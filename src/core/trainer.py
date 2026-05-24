"""QThread-based training worker.  Validation set is NEVER used during training."""
import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from qtpy.QtCore import QThread, Signal

from .dataset import FrothDataset, collect_samples, build_transform
from .model_builder import build_model, count_trainable_params


OPTIMIZERS = {
    "Adam":  torch.optim.Adam,
    "SGD":   torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
}

REGRESSION_LOSSES = {
    "MSE":      nn.MSELoss(),
    "MAE":      nn.L1Loss(),
    "Huber":    nn.HuberLoss(),
    "SmoothL1": nn.SmoothL1Loss(),
}


class TrainingWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)          # epoch, total_epochs
    epoch_metrics = Signal(dict)         # {epoch, train_loss, test_loss, ...}
    finished = Signal(dict)              # full result dict
    error = Signal(str)

    def __init__(self, config: dict, run_params: dict = None, run_id: str = "run"):
        super().__init__()
        self.config = config
        self.run_params = run_params or {}
        self.run_id = run_id
        self._stop = False

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------------
    def run(self):
        try:
            result = self._train()
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())

    def _train(self) -> dict:
        cfg = self.config
        t_cfg = cfg["training"]
        prep_cfg = cfg["preprocessing"]
        model_cfg = cfg["model"]
        experiments = cfg["experiments"]
        class_cfg = cfg.get("classification", {"enabled": False, "classes": []})

        is_cls = class_cfg.get("enabled", False)
        classes = class_cfg.get("classes", []) if is_cls else []
        num_outputs = len(classes) if is_cls else 1

        # Resolve grid-search overrides
        resolved = self._resolve_params(t_cfg, self.run_params)

        epochs = int(t_cfg.get("epochs", 50))
        batch_size = int(resolved["batch_size"])
        lr = float(resolved["learning_rate"])
        opt_name = resolved["optimizer"]
        wd = float(resolved["weight_decay"])
        loss_name = resolved["loss"]
        num_workers = int(t_cfg.get("num_workers", 0))

        # Architecture and transfer settings now live in t_cfg (moved from model_cfg)
        arch_name       = resolved.get("architecture", "ResNet-50")
        pretrained      = bool(t_cfg.get("pretrained", True))
        freeze_backbone = bool(t_cfg.get("freeze_backbone", False))
        unfreeze_last_n = int(t_cfg.get("unfreeze_last_n", 0))
        dropout_head    = float(t_cfg.get("dropout_head", 0.5))

        eff_model_cfg = dict(model_cfg)
        if model_cfg.get("mode") == "transfer":
            eff_model_cfg["transfer"] = {
                "architecture": arch_name,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
                "unfreeze_last_n": unfreeze_last_n,
                "dropout": dropout_head,
            }

        mode_label = "classification" if is_cls else "regression"
        self.log.emit(f"[{self.run_id}] Building model… (arch={arch_name}, mode={mode_label}, outputs={num_outputs})")
        model = build_model(eff_model_cfg, num_outputs=num_outputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        n_params = count_trainable_params(model)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # speeds up fixed-size inputs
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.log.emit(f"[{self.run_id}] GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        self.log.emit(f"[{self.run_id}] Device: {device} | Trainable params: {n_params:,}")

        if is_cls:
            class_names = [c.get("name", str(i)) for i, c in enumerate(classes)]
            self.log.emit(f"[{self.run_id}] Classes: {class_names}")

        # Datasets
        train_tf = build_transform(prep_cfg, augment=True)
        eval_tf = build_transform(prep_cfg, augment=False)

        # "validation" split → monitored during training (early stopping, LR scheduling)
        # "test" split      → held out, evaluated exactly once after training ends
        train_samples = collect_samples(experiments, "train",      class_cfg)
        val_samples   = collect_samples(experiments, "validation", class_cfg)  # monitored
        test_samples  = collect_samples(experiments, "test",       class_cfg)  # held out

        if not train_samples:
            raise ValueError("No training images found. Check experiment folder paths.")
        if not val_samples:
            self.log.emit(f"[{self.run_id}] WARNING: No validation images found. Using train loss for early stopping.")

        self.log.emit(
            f"[{self.run_id}] Samples — train: {len(train_samples)}, "
            f"val: {len(val_samples)} (monitored), test: {len(test_samples)} (held out)"
        )

        train_ds = FrothDataset(train_samples, train_tf)
        val_ds   = FrothDataset(val_samples,   eval_tf) if val_samples  else None  # monitored
        test_ds  = FrothDataset(test_samples,  eval_tf) if test_samples else None  # held out

        pin_mem = device.type == "cuda"
        use_amp = t_cfg.get("use_amp", False) and device.type == "cuda"
        if use_amp:
            self.log.emit(f"[{self.run_id}] Automatic Mixed Precision (AMP) enabled — float16 forward pass")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_mem,
                                  persistent_workers=(num_workers > 0))
        # val_loader is used DURING the training loop (early stopping / LR scheduling)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_mem,
                                persistent_workers=(num_workers > 0)) if val_ds else None
        # test_loader is NEVER touched during the training loop
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_mem,
                                 persistent_workers=(num_workers > 0)) if test_ds else None

        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Loss
        if is_cls:
            # Compute class weights from training set (inverse frequency)
            counts = [0] * num_outputs
            for _, lbl in train_samples:
                counts[int(lbl)] += 1
            weights = torch.tensor(
                [1.0 / (c + 1e-6) for c in counts], dtype=torch.float32
            )
            weights = weights / weights.sum() * num_outputs
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
            self.log.emit(
                f"[{self.run_id}] CrossEntropyLoss | class counts: {counts} | "
                f"weights: {[f'{w:.3f}' for w in weights.tolist()]}"
            )
        else:
            criterion = REGRESSION_LOSSES.get(loss_name, nn.MSELoss())

        # Optimizer
        opt_cls = OPTIMIZERS.get(opt_name, torch.optim.Adam)
        if opt_name == "SGD":
            momentum = float(t_cfg.get("momentum", 0.9))
            optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        else:
            optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        sched_cfg = t_cfg.get("lr_scheduler", {})
        scheduler = _build_scheduler(optimizer, sched_cfg)

        # Early stopping
        es_cfg = t_cfg.get("early_stopping", {})
        es_enabled = es_cfg.get("enabled", True)
        es_patience = int(es_cfg.get("patience", 15))
        es_min_delta = float(es_cfg.get("min_delta", 1e-4))

        # Output directory
        out_dir = t_cfg.get("output_dir", "./results")
        run_dir = os.path.join(out_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        best_ckpt = os.path.join(run_dir, "best_model.pt")

        # Training loop
        # val_loader (validation split) is monitored each epoch for early stopping / scheduling
        # test_loader (test split) is NEVER touched here
        train_history, val_history = [], []
        best_monitor = math.inf   # lower-is-better (loss); for cls we negate accuracy
        no_improve = 0
        best_epoch = 0
        start_time = time.time()

        _train_fn = _train_epoch_cls if is_cls else _train_epoch_reg
        _eval_fn  = _eval_epoch_cls  if is_cls else _eval_epoch_reg

        for epoch in range(1, epochs + 1):
            if self._stop:
                self.log.emit(f"[{self.run_id}] Stopped by user at epoch {epoch}.")
                break

            train_loss = _train_fn(model, train_loader, optimizer, criterion, device, scaler)
            train_history.append(train_loss)

            monitor_val = train_loss
            val_metrics_ep = {}
            if val_loader:
                val_metrics_ep = _eval_fn(model, val_loader, criterion, device, num_outputs)
                val_history.append(val_metrics_ep["loss"])
                if is_cls:
                    monitor_val = -val_metrics_ep.get("accuracy", 0.0)
                else:
                    monitor_val = val_metrics_ep["loss"]
            else:
                val_history.append(None)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_val)
            elif scheduler is not None:
                scheduler.step()

            # Save best checkpoint
            if monitor_val < best_monitor - es_min_delta:
                best_monitor = monitor_val
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                no_improve += 1

            current_lr = optimizer.param_groups[0]["lr"]
            self.progress.emit(epoch, epochs)

            metrics_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics_ep.get("loss"),
                "lr": current_lr,
            }
            if is_cls:
                metrics_payload["val_accuracy"] = val_metrics_ep.get("accuracy")
                metrics_payload["val_f1"] = val_metrics_ep.get("f1")
                self.log.emit(
                    f"[{self.run_id}] Ep {epoch}/{epochs} | "
                    f"train={train_loss:.4f} | "
                    f"val_loss={val_metrics_ep.get('loss', '-')!s:.6} | "
                    f"val_acc={val_metrics_ep.get('accuracy', 0):.3f} | "
                    f"lr={current_lr:.2e}"
                )
            else:
                metrics_payload["val_rmse"] = val_metrics_ep.get("rmse")
                metrics_payload["val_mae"] = val_metrics_ep.get("mae")
                self.log.emit(
                    f"[{self.run_id}] Ep {epoch}/{epochs} | "
                    f"train={train_loss:.4f} | "
                    f"val_loss={val_metrics_ep.get('loss', '-')!s:.6} | "
                    f"lr={current_lr:.2e}"
                )

            self.epoch_metrics.emit(metrics_payload)

            if es_enabled and no_improve >= es_patience:
                self.log.emit(f"[{self.run_id}] Early stopping at epoch {epoch} (no improve for {es_patience} epochs).")
                break

        elapsed = time.time() - start_time

        # Load best checkpoint
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        # Final eval on validation set (the monitoring split)
        final_val_metrics = {}
        if val_loader:
            final_val_metrics = _eval_fn(model, val_loader, criterion, device, num_outputs)

        # Final eval on test set (held-out) — done ONCE, never during training loop
        final_test_metrics = {}
        if test_loader:
            self.log.emit(f"[{self.run_id}] Final evaluation on held-out test set…")
            final_test_metrics = _eval_fn(model, test_loader, criterion, device, num_outputs)
            if is_cls:
                self.log.emit(
                    f"[{self.run_id}] TEST acc={final_test_metrics.get('accuracy', 0):.4f} | "
                    f"F1={final_test_metrics.get('f1', 0):.4f}"
                )
            else:
                self.log.emit(
                    f"[{self.run_id}] TEST RMSE={final_test_metrics.get('rmse', 0):.4f} | "
                    f"MAE={final_test_metrics.get('mae', 0):.4f} | "
                    f"R²={final_test_metrics.get('r2', 0):.4f}"
                )

        return {
            "run_id": self.run_id,
            "mode": mode_label,
            "class_names": [c.get("name", str(i)) for i, c in enumerate(classes)] if is_cls else [],
            "params": {**resolved, **self.run_params},
            "train_history": train_history,
            "val_history": val_history,          # validation (monitoring) loss per epoch
            "best_epoch": best_epoch,
            "final_val_metrics": final_val_metrics,   # validation split (monitored)
            "final_test_metrics": final_test_metrics, # test split (held out)
            "elapsed_seconds": elapsed,
            "checkpoint_path": best_ckpt if os.path.exists(best_ckpt) else "",
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
        }

    @staticmethod
    def _resolve_params(t_cfg: dict, overrides: dict) -> dict:
        def pick(key, default):
            if key in overrides:
                return overrides[key]
            entry = t_cfg.get(key, {})
            if isinstance(entry, dict):
                return entry.get("value", default)
            return entry

        return {
            "batch_size":    pick("batch_size",    "32"),
            "learning_rate": pick("learning_rate", "0.001"),
            "optimizer":     pick("optimizer",     "Adam"),
            "weight_decay":  pick("weight_decay",  "1e-4"),
            "loss":          pick("loss",           "MSE"),
            "architecture":  pick("architecture",   "ResNet-50"),
        }


# ---------------------------------------------------------------------------

def _train_epoch_reg(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def _eval_epoch_reg(model, loader, criterion, device, _num_outputs=1):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, labels)
            total_loss += loss.item() * len(imgs)
            all_preds.extend(preds.cpu().squeeze(1).numpy())
            all_labels.extend(labels.cpu().squeeze(1).numpy())

    p = np.array(all_preds)
    y = np.array(all_labels)
    n = len(y)
    mse = float(np.mean((p - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - y)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "loss": total_loss / n if n > 0 else 0.0,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _train_epoch_cls(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def _eval_epoch_cls(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(imgs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    p = np.array(all_preds)
    y = np.array(all_labels)
    n = len(y)
    accuracy = float((p == y).mean()) if n > 0 else 0.0
    f1 = _macro_f1(p, y, num_classes)

    return {
        "loss": total_loss / n if n > 0 else 0.0,
        "accuracy": accuracy,
        "f1": f1,
        "predictions": all_preds,    # stored for confusion matrix
        "true_labels": all_labels,
    }


def _macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def _build_scheduler(optimizer, cfg: dict):
    stype = cfg.get("type", "None")
    if stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.get("t_max", 50))
        )
    if stype == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(cfg.get("patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
    return None
