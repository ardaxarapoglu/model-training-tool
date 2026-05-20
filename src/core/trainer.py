"""QThread-based training worker.  Validation set is NEVER used during training."""
import os
import time
import math

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

LOSSES = {
    "MSE":   nn.MSELoss(),
    "MAE":   nn.L1Loss(),
    "Huber": nn.HuberLoss(),
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

        # Resolve grid-search overrides
        resolved = self._resolve_params(t_cfg, self.run_params)

        epochs = int(t_cfg.get("epochs", 50))
        batch_size = int(resolved["batch_size"])
        lr = float(resolved["learning_rate"])
        opt_name = resolved["optimizer"]
        wd = float(resolved["weight_decay"])
        loss_name = resolved["loss"]
        num_workers = int(t_cfg.get("num_workers", 0))

        # Model (allow architecture override from grid-search)
        eff_model_cfg = dict(model_cfg)
        if "architecture" in self.run_params and model_cfg.get("mode") == "transfer":
            eff_model_cfg = dict(model_cfg)
            eff_model_cfg["transfer"] = dict(model_cfg.get("transfer", {}))
            eff_model_cfg["transfer"]["architecture"] = self.run_params["architecture"]

        self.log.emit(f"[{self.run_id}] Building model…")
        model = build_model(eff_model_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        n_params = count_trainable_params(model)
        self.log.emit(f"[{self.run_id}] Device: {device} | Trainable params: {n_params:,}")

        # Datasets
        train_tf = build_transform(prep_cfg, augment=True)
        eval_tf = build_transform(prep_cfg, augment=False)

        train_samples = collect_samples(experiments, "train")
        test_samples = collect_samples(experiments, "test")
        val_samples = collect_samples(experiments, "validation")

        if not train_samples:
            raise ValueError("No training images found. Check experiment folder paths.")
        if not test_samples:
            self.log.emit(f"[{self.run_id}] WARNING: No test images found. Using train loss for early stopping.")

        self.log.emit(
            f"[{self.run_id}] Samples — train: {len(train_samples)}, "
            f"test: {len(test_samples)}, val: {len(val_samples)} (held out)"
        )

        train_ds = FrothDataset(train_samples, train_tf)
        test_ds = FrothDataset(test_samples, eval_tf) if test_samples else None
        val_ds = FrothDataset(val_samples, eval_tf) if val_samples else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds else None
        # val_loader is never used during training loop
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None

        # Loss
        criterion = LOSSES.get(loss_name, nn.MSELoss())

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
        train_history, test_history = [], []
        best_test_loss = math.inf
        no_improve = 0
        best_epoch = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self._stop:
                self.log.emit(f"[{self.run_id}] Stopped by user at epoch {epoch}.")
                break

            train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
            train_history.append(train_loss)

            monitor_loss = train_loss
            test_metrics_ep = {}
            if test_loader:
                test_metrics_ep = _eval_epoch(model, test_loader, criterion, device)
                test_history.append(test_metrics_ep["loss"])
                monitor_loss = test_metrics_ep["loss"]
            else:
                test_history.append(None)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_loss)
            elif scheduler is not None:
                scheduler.step()

            # Save best
            if monitor_loss < best_test_loss - es_min_delta:
                best_test_loss = monitor_loss
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
                "test_loss": test_metrics_ep.get("loss"),
                "test_rmse": test_metrics_ep.get("rmse"),
                "test_mae": test_metrics_ep.get("mae"),
                "lr": current_lr,
            }
            self.epoch_metrics.emit(metrics_payload)
            self.log.emit(
                f"[{self.run_id}] Ep {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_metrics_ep.get('loss', '-')!s:.6} | "
                f"lr={current_lr:.2e}"
            )

            if es_enabled and no_improve >= es_patience:
                self.log.emit(f"[{self.run_id}] Early stopping at epoch {epoch} (no improve for {es_patience} epochs).")
                break

        elapsed = time.time() - start_time

        # Load best checkpoint for final eval
        final_test_metrics = {}
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
        if test_loader:
            final_test_metrics = _eval_epoch(model, test_loader, criterion, device)

        # Validation eval — done ONCE after training is fully complete
        # (never used for early stopping or hyperparameter selection)
        final_val_metrics = {}
        if val_loader:
            self.log.emit(f"[{self.run_id}] Final evaluation on validation set (held-out)…")
            final_val_metrics = _eval_epoch(model, val_loader, criterion, device)
            self.log.emit(
                f"[{self.run_id}] VAL RMSE={final_val_metrics.get('rmse', '?'):.4f} | "
                f"MAE={final_val_metrics.get('mae', '?'):.4f} | "
                f"R²={final_val_metrics.get('r2', '?'):.4f}"
            )

        return {
            "run_id": self.run_id,
            "params": {**resolved, **self.run_params},
            "train_history": train_history,
            "test_history": test_history,
            "best_epoch": best_epoch,
            "final_test_metrics": final_test_metrics,
            "final_val_metrics": final_val_metrics,
            "elapsed_seconds": elapsed,
            "checkpoint_path": best_ckpt if os.path.exists(best_ckpt) else "",
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "n_val": len(val_samples),
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
            "batch_size": pick("batch_size", "32"),
            "learning_rate": pick("learning_rate", "0.001"),
            "optimizer": pick("optimizer", "Adam"),
            "weight_decay": pick("weight_decay", "1e-4"),
            "loss": pick("loss", "MSE"),
        }


# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, device):
    import numpy as np
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
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
