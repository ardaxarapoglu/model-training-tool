"""Training panel: hyperparameters, grid-search, run/stop controls, live log."""
import os

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QPushButton, QProgressBar, QTextEdit, QSplitter,
    QScrollArea, QFrame, QFileDialog, QSizePolicy, QTabWidget,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont, QColor, QPalette


class _ParamRow(QWidget):
    """Single parameter row with value input + optional multi-value field (for grid search)."""

    def __init__(self, label: str, default: str, tooltip: str = ""):
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label)
        lbl.setMinimumWidth(130)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(lbl)

        self.edit_value = QLineEdit(default)
        self.edit_value.setMaximumWidth(120)
        if tooltip:
            self.edit_value.setToolTip(tooltip)
        hl.addWidget(self.edit_value)

        sep = QLabel("|")
        sep.setStyleSheet("color:#ccc;margin:0 4px;")
        hl.addWidget(sep)

        self.chk_grid = QCheckBox("Grid:")
        self.chk_grid.setToolTip("Enable multiple values for grid search (comma-separated)")
        self.chk_grid.setMaximumWidth(55)
        hl.addWidget(self.chk_grid)

        self.edit_grid = QLineEdit()
        self.edit_grid.setPlaceholderText("e.g., 0.01, 0.001, 0.0001")
        self.edit_grid.setEnabled(False)
        hl.addWidget(self.edit_grid)

        self.chk_grid.toggled.connect(self.edit_grid.setEnabled)
        hl.addStretch()

    def get_entry(self) -> dict:
        return {
            "value": self.edit_value.text().strip(),
            "values": self.edit_grid.text().strip(),
            "use_grid": self.chk_grid.isChecked(),
        }

    def set_entry(self, entry):
        if isinstance(entry, dict):
            self.edit_value.setText(str(entry.get("value", "")))
            self.edit_grid.setText(str(entry.get("values", "")))
            self.chk_grid.setChecked(bool(entry.get("use_grid", False)))
        else:
            self.edit_value.setText(str(entry))

    def set_grid_visible(self, visible: bool):
        self.chk_grid.setVisible(visible)
        self.edit_grid.setVisible(visible)


class TrainingPanel(QWidget):
    run_requested = Signal()

    def __init__(self):
        super().__init__()
        self._worker = None
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # -------- Left: config --------
        left = QWidget()
        left.setMaximumWidth(560)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        cfg_container = QWidget()
        cfg_v = QVBoxLayout(cfg_container)
        cfg_v.setSpacing(8)
        scroll.setWidget(cfg_container)
        lv.addWidget(scroll)

        # Grid search toggle banner
        gs_grp = QGroupBox("Grid Search")
        gs_v = QVBoxLayout(gs_grp)
        self.chk_grid_search = QCheckBox(
            "Enable grid search  (train all combinations of highlighted parameters)"
        )
        self.chk_grid_search.toggled.connect(self._on_grid_search_toggled)
        gs_v.addWidget(self.chk_grid_search)
        gs_info = QLabel(
            "When enabled, enable the  Grid:  checkbox on any parameter below to "
            "include multiple values.  The tool will train every combination and "
            "rank results by test RMSE.\n"
            "⚠  Validation set is NEVER used during grid search — "
            "it is evaluated once for the final report only."
        )
        gs_info.setWordWrap(True)
        gs_info.setStyleSheet("color:#555;font-size:11px;")
        gs_v.addWidget(gs_info)
        cfg_v.addWidget(gs_grp)

        # Hyperparameter rows
        params_grp = QGroupBox("Hyperparameters")
        params_form = QFormLayout(params_grp)
        params_form.setLabelAlignment(Qt.AlignRight)
        cfg_v.addWidget(params_grp)

        # Epochs (no grid search — always single value)
        ep_row = QHBoxLayout()
        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 9999)
        self.sp_epochs.setValue(50)
        ep_row.addWidget(self.sp_epochs)
        ep_row.addStretch()
        params_form.addRow("Epochs:", ep_row)

        self.row_bs   = _ParamRow("Batch size",     "32",    "Images per training batch")
        self.row_lr   = _ParamRow("Learning rate",  "0.001", "Initial learning rate")
        self.row_opt  = _ParamRow("Optimizer",      "Adam",  "Adam | SGD | AdamW | RMSprop")
        self.row_wd   = _ParamRow("Weight decay",   "1e-4",  "L2 regularization coefficient")
        self.row_loss = _ParamRow("Loss function",  "MSE",   "MSE | MAE | Huber | SmoothL1")
        for row in (self.row_bs, self.row_lr, self.row_opt, self.row_wd, self.row_loss):
            params_form.addRow(row)
            row.set_grid_visible(False)   # hidden until grid search enabled

        # SGD momentum (always single value)
        mom_row = QHBoxLayout()
        self.sp_momentum = QDoubleSpinBox()
        self.sp_momentum.setRange(0.0, 1.0)
        self.sp_momentum.setSingleStep(0.01)
        self.sp_momentum.setValue(0.9)
        self.sp_momentum.setDecimals(2)
        self.sp_momentum.setMaximumWidth(100)
        mom_row.addWidget(self.sp_momentum)
        mom_row.addWidget(QLabel("(used for SGD only)"))
        mom_row.addStretch()
        params_form.addRow("SGD momentum:", mom_row)

        # LR Scheduler
        sched_grp = QGroupBox("LR Scheduler")
        cfg_v.addWidget(sched_grp)
        sched_form = QFormLayout(sched_grp)
        sched_form.setLabelAlignment(Qt.AlignRight)

        self.cmb_sched = QComboBox()
        for s in ("None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"):
            self.cmb_sched.addItem(s)
        self.cmb_sched.setCurrentText("StepLR")
        sched_form.addRow("Type:", self.cmb_sched)

        self.sp_step_size = QSpinBox()
        self.sp_step_size.setRange(1, 999)
        self.sp_step_size.setValue(10)
        sched_form.addRow("Step size:", self.sp_step_size)

        self.sp_gamma = QDoubleSpinBox()
        self.sp_gamma.setRange(0.01, 1.0)
        self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setDecimals(2)
        self.sp_gamma.setValue(0.5)
        sched_form.addRow("Gamma:", self.sp_gamma)

        self.sp_tmax = QSpinBox()
        self.sp_tmax.setRange(1, 9999)
        self.sp_tmax.setValue(50)
        sched_form.addRow("T_max (cosine):", self.sp_tmax)

        self.sp_plat_patience = QSpinBox()
        self.sp_plat_patience.setRange(1, 999)
        self.sp_plat_patience.setValue(5)
        sched_form.addRow("Plateau patience:", self.sp_plat_patience)

        self.sp_min_lr = QDoubleSpinBox()
        self.sp_min_lr.setRange(0.0, 0.1)
        self.sp_min_lr.setSingleStep(1e-6)
        self.sp_min_lr.setDecimals(8)
        self.sp_min_lr.setValue(1e-6)
        sched_form.addRow("Min LR:", self.sp_min_lr)

        # Early stopping
        es_grp = QGroupBox("Early Stopping")
        cfg_v.addWidget(es_grp)
        es_form = QFormLayout(es_grp)
        es_form.setLabelAlignment(Qt.AlignRight)

        self.chk_early_stop = QCheckBox("Enable")
        self.chk_early_stop.setChecked(True)
        es_form.addRow("", self.chk_early_stop)

        self.sp_es_patience = QSpinBox()
        self.sp_es_patience.setRange(1, 999)
        self.sp_es_patience.setValue(15)
        es_form.addRow("Patience (epochs):", self.sp_es_patience)

        self.sp_es_delta = QDoubleSpinBox()
        self.sp_es_delta.setRange(0.0, 1.0)
        self.sp_es_delta.setSingleStep(1e-4)
        self.sp_es_delta.setDecimals(6)
        self.sp_es_delta.setValue(1e-4)
        es_form.addRow("Min delta:", self.sp_es_delta)

        # Output
        out_grp = QGroupBox("Output")
        cfg_v.addWidget(out_grp)
        out_form = QFormLayout(out_grp)
        out_form.setLabelAlignment(Qt.AlignRight)

        out_row = QHBoxLayout()
        self.edit_out_dir = QLineEdit("./results")
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.edit_out_dir)
        out_row.addWidget(btn_browse_out)
        out_form.addRow("Output directory:", out_row)

        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(0, 32)
        self.sp_workers.setValue(0)
        self.sp_workers.setToolTip("0 = run data loading in main process (safe on Windows)")
        out_form.addRow("DataLoader workers:", self.sp_workers)

        cfg_v.addStretch()
        splitter.addWidget(left)

        # -------- Right: run & monitor --------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(6)

        ctrl_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Training")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#388E3C;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_run.clicked.connect(self._on_run)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(36)
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#f44336;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#c62828;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        ctrl_row.addWidget(self.btn_run)
        ctrl_row.addWidget(self.btn_stop)
        rv.addLayout(ctrl_row)

        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight:bold;")
        rv.addWidget(self.lbl_status)

        # Progress
        self.lbl_run_progress = QLabel("Run 0 / 0")
        rv.addWidget(self.lbl_run_progress)
        self.pbar_runs = QProgressBar()
        self.pbar_runs.setTextVisible(True)
        rv.addWidget(self.pbar_runs)

        self.lbl_epoch_progress = QLabel("Epoch 0 / 0")
        rv.addWidget(self.lbl_epoch_progress)
        self.pbar_epoch = QProgressBar()
        rv.addWidget(self.pbar_epoch)

        # Live metrics
        metrics_grp = QGroupBox("Current Metrics")
        metrics_h = QHBoxLayout(metrics_grp)
        self.lbl_train_loss = _metric_label("Train Loss", "—")
        self.lbl_test_loss  = _metric_label("Test Loss",  "—")
        self.lbl_test_rmse  = _metric_label("Test RMSE",  "—")
        self.lbl_test_mae   = _metric_label("Test MAE",   "—")
        for w in (self.lbl_train_loss, self.lbl_test_loss, self.lbl_test_rmse, self.lbl_test_mae):
            metrics_h.addWidget(w)
        rv.addWidget(metrics_grp)

        # Log
        log_grp = QGroupBox("Training Log")
        log_v = QVBoxLayout(log_grp)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        log_btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("Clear")
        btn_clear_log.clicked.connect(self.log_edit.clear)
        log_btn_row.addStretch()
        log_btn_row.addWidget(btn_clear_log)
        log_v.addLayout(log_btn_row)
        log_v.addWidget(self.log_edit)
        rv.addWidget(log_grp)

        splitter.addWidget(right)
        splitter.setSizes([500, 700])

    # ---------------------------------------------------------------- grid search toggle
    def _on_grid_search_toggled(self, enabled: bool):
        for row in (self.row_bs, self.row_lr, self.row_opt, self.row_wd, self.row_loss):
            row.set_grid_visible(enabled)

    # ---------------------------------------------------------------- output dir browse
    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.edit_out_dir.setText(path)

    # ---------------------------------------------------------------- run / stop
    def _on_run(self):
        self.run_requested.emit()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self.lbl_status.setText("Stopping…")
            self.btn_stop.setEnabled(False)

    def start_training(self, full_config: dict):
        from ..core.trainer import TrainingWorker
        from ..core.grid_search import GridSearchWorker

        t_cfg = full_config["training"]
        use_grid = t_cfg.get("grid_search_enabled", False)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_edit.clear()
        self._reset_metrics()

        if use_grid:
            worker = GridSearchWorker(full_config)
            worker.run_log.connect(self._on_log)
            worker.run_started.connect(self._on_gs_run_started)
            worker.run_progress.connect(self._on_gs_progress)
            worker.run_finished.connect(self._on_gs_run_finished)
            worker.all_done.connect(self._on_all_done)
            worker.error.connect(self._on_error)
            self.lbl_status.setText("Grid search running…")
        else:
            worker = TrainingWorker(full_config, run_id="single_run")
            worker.log.connect(self._on_log)
            worker.progress.connect(lambda ep, tot: self._on_epoch_progress(ep, tot))
            worker.epoch_metrics.connect(self._on_epoch_metrics)
            worker.finished.connect(self._on_single_done)
            worker.error.connect(self._on_error)
            self.lbl_status.setText("Training…")
            self.pbar_runs.setRange(0, 1)
            self.pbar_runs.setValue(0)
            self.lbl_run_progress.setText("Run 1 / 1")

        self._worker = worker
        worker.start()

    # ---------------------------------------------------------------- worker slots
    def _on_log(self, msg: str):
        self.log_edit.append(msg)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )

    def _on_epoch_progress(self, epoch: int, total: int):
        self.lbl_epoch_progress.setText(f"Epoch {epoch} / {total}")
        self.pbar_epoch.setRange(0, total)
        self.pbar_epoch.setValue(epoch)

    def _on_epoch_metrics(self, m: dict):
        self.lbl_train_loss.findChild(QLabel, "value").setText(f"{m.get('train_loss', 0):.4f}")
        tl = m.get("test_loss")
        self.lbl_test_loss.findChild(QLabel, "value").setText(f"{tl:.4f}" if tl else "—")
        tr = m.get("test_rmse")
        self.lbl_test_rmse.findChild(QLabel, "value").setText(f"{tr:.4f}" if tr else "—")
        tm = m.get("test_mae")
        self.lbl_test_mae.findChild(QLabel, "value").setText(f"{tm:.4f}" if tm else "—")

    def _on_gs_run_started(self, run_num: int, total: int, params: dict):
        self.lbl_run_progress.setText(f"Run {run_num} / {total}")
        self.pbar_runs.setRange(0, total)
        self.pbar_runs.setValue(run_num - 1)
        self.pbar_epoch.setValue(0)
        self._reset_metrics()

    def _on_gs_progress(self, run_num, total_runs, epoch, total_epochs):
        self._on_epoch_progress(epoch, total_epochs)

    def _on_gs_run_finished(self, run_num: int, result: dict):
        self.pbar_runs.setValue(run_num)

    def _on_single_done(self, result: dict):
        self.pbar_runs.setValue(1)
        self._training_finished([result])

    def _on_all_done(self, results: list):
        self._training_finished(results)

    def _on_error(self, msg: str):
        self.log_edit.append(f"\n[ERROR]\n{msg}")
        self.lbl_status.setText("Error — see log")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _training_finished(self, results: list):
        self.lbl_status.setText(f"Done  ({len(results)} run(s))")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Emit results to main window for results panel
        mw = self.window()
        if hasattr(mw, "receive_training_results"):
            mw.receive_training_results(results)

    def _reset_metrics(self):
        for w in (self.lbl_train_loss, self.lbl_test_loss, self.lbl_test_rmse, self.lbl_test_mae):
            vl = w.findChild(QLabel, "value")
            if vl:
                vl.setText("—")

    # ---------------------------------------------------------------- public API
    def get_config(self) -> dict:
        return {
            "epochs": self.sp_epochs.value(),
            "batch_size": self.row_bs.get_entry(),
            "learning_rate": self.row_lr.get_entry(),
            "optimizer": self.row_opt.get_entry(),
            "weight_decay": self.row_wd.get_entry(),
            "loss": self.row_loss.get_entry(),
            "momentum": self.sp_momentum.value(),
            "lr_scheduler": {
                "type": self.cmb_sched.currentText(),
                "step_size": self.sp_step_size.value(),
                "gamma": self.sp_gamma.value(),
                "t_max": self.sp_tmax.value(),
                "patience": self.sp_plat_patience.value(),
                "min_lr": self.sp_min_lr.value(),
            },
            "early_stopping": {
                "enabled": self.chk_early_stop.isChecked(),
                "patience": self.sp_es_patience.value(),
                "min_delta": self.sp_es_delta.value(),
            },
            "grid_search_enabled": self.chk_grid_search.isChecked(),
            "output_dir": self.edit_out_dir.text(),
            "num_workers": self.sp_workers.value(),
        }

    def set_config(self, cfg: dict):
        self.sp_epochs.setValue(int(cfg.get("epochs", 50)))
        self.row_bs.set_entry(cfg.get("batch_size", "32"))
        self.row_lr.set_entry(cfg.get("learning_rate", "0.001"))
        self.row_opt.set_entry(cfg.get("optimizer", "Adam"))
        self.row_wd.set_entry(cfg.get("weight_decay", "1e-4"))
        self.row_loss.set_entry(cfg.get("loss", "MSE"))
        self.sp_momentum.setValue(float(cfg.get("momentum", 0.9)))

        sched = cfg.get("lr_scheduler", {})
        idx = self.cmb_sched.findText(sched.get("type", "StepLR"))
        if idx >= 0:
            self.cmb_sched.setCurrentIndex(idx)
        self.sp_step_size.setValue(int(sched.get("step_size", 10)))
        self.sp_gamma.setValue(float(sched.get("gamma", 0.5)))
        self.sp_tmax.setValue(int(sched.get("t_max", 50)))
        self.sp_plat_patience.setValue(int(sched.get("patience", 5)))
        self.sp_min_lr.setValue(float(sched.get("min_lr", 1e-6)))

        es = cfg.get("early_stopping", {})
        self.chk_early_stop.setChecked(bool(es.get("enabled", True)))
        self.sp_es_patience.setValue(int(es.get("patience", 15)))
        self.sp_es_delta.setValue(float(es.get("min_delta", 1e-4)))

        self.chk_grid_search.setChecked(bool(cfg.get("grid_search_enabled", False)))
        self.edit_out_dir.setText(str(cfg.get("output_dir", "./results")))
        self.sp_workers.setValue(int(cfg.get("num_workers", 0)))


# ---------------------------------------------------------------- helpers
def _metric_label(title: str, init: str) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(4, 2, 4, 2)
    t = QLabel(title)
    t.setAlignment(Qt.AlignCenter)
    t.setStyleSheet("font-size:10px;color:#666;")
    val = QLabel(init)
    val.setObjectName("value")
    val.setAlignment(Qt.AlignCenter)
    val.setFont(QFont("Arial", 14, QFont.Bold))
    v.addWidget(t)
    v.addWidget(val)
    return w
