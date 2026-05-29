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

from ..core.model_builder import ARCHITECTURES


class _ParamRow(QWidget):
    """Single parameter row with value input + optional multi-value field (for grid search).

    When *choices* is provided a QComboBox is used instead of a free-text QLineEdit,
    preventing typos in fields like Architecture, Optimizer, and Loss function.
    """

    def __init__(self, label: str, default: str, tooltip: str = "",
                 choices: list = None):
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 2, 0, 2)

        lbl = QLabel(label)
        lbl.setMinimumWidth(130)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hl.addWidget(lbl)

        self._has_choices = choices is not None
        if self._has_choices:
            self.edit_value = QComboBox()
            self.edit_value.addItems(choices)
            self.edit_value.setCurrentText(default)
            self.edit_value.setMaximumWidth(180)
        else:
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
        value = (self.edit_value.currentText()
                 if self._has_choices
                 else self.edit_value.text().strip())
        return {
            "value": value,
            "values": self.edit_grid.text().strip(),
            "use_grid": self.chk_grid.isChecked(),
        }

    def set_entry(self, entry):
        if isinstance(entry, dict):
            val = str(entry.get("value", ""))
            if self._has_choices:
                self.edit_value.setCurrentText(val)
            else:
                self.edit_value.setText(val)
            self.edit_grid.setText(str(entry.get("values", "")))
            self.chk_grid.setChecked(bool(entry.get("use_grid", False)))
        else:
            val = str(entry)
            if self._has_choices:
                self.edit_value.setCurrentText(val)
            else:
                self.edit_value.setText(val)

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
        self.chk_grid_search.setToolTip(
            "Automatically train one model for every combination of the parameters\n"
            "you mark with 'Grid:' below (e.g. 3 LRs × 2 batch sizes = 6 runs).\n"
            "All runs share the same epochs, scheduler, and architecture unless those\n"
            "are also included in the grid. Results are ranked by test RMSE or accuracy."
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

        # ---- Architecture (Transfer Learning) ----
        arch_grp = QGroupBox("Architecture  (Transfer Learning)")
        arch_grp.setToolTip(
            "Settings used when 'Transfer Learning' is selected in the ③ Model tab.\n"
            "Ignored when 'Train from Scratch' is selected."
        )
        arch_form = QFormLayout(arch_grp)
        arch_form.setLabelAlignment(Qt.AlignRight)
        cfg_v.addWidget(arch_grp)

        self.row_arch = _ParamRow("Architecture", "ResNet-50",
            "The CNN backbone to use for feature extraction.\n"
            "• ResNet-50 / 101   – solid default; fast to train.\n"
            "• EfficientNet-B0   – small and fast.\n"
            "• EfficientNet-B3/4 – larger, often more accurate, slower.\n"
            "• MobileNet-V3      – very fast, lower capacity.\n"
            "• DenseNet-121      – strong feature reuse.\n"
            "Grid: enter multiple names (e.g. ResNet-50,EfficientNet-B0) to sweep.",
            choices=list(ARCHITECTURES.keys()),
        )
        arch_form.addRow(self.row_arch)

        self.chk_pretrained = QCheckBox("Use pretrained ImageNet weights")
        self.chk_pretrained.setChecked(True)
        self.chk_pretrained.setToolTip(
            "Load weights trained on ImageNet as the starting point.\n"
            "Strongly recommended — the network already knows general visual features\n"
            "and converges much faster with far fewer images."
        )
        arch_form.addRow("", self.chk_pretrained)

        self.row_freeze = _ParamRow(
            "Freeze backbone", "True",
            "Lock all backbone weights so only the prediction head is updated.\n"
            "True  → fast training, good when images are few (< ~500 per class).\n"
            "False → full fine-tuning; slower, needs more data.\n"
            "Combine with 'Unfreeze last N' for a middle ground.\n"
            "Grid: True,False to compare both strategies.",
            choices=["True", "False"],
        )
        arch_form.addRow(self.row_freeze)

        self.row_unfreeze = _ParamRow(
            "Unfreeze last N", "0",
            "Number of backbone parameter tensors (from the end) to unfreeze for fine-tuning.\n"
            "0 = fully respect the Freeze Backbone setting.\n"
            "5 = fine-tune the last 5 tensors (domain-specific features) — recommended.\n"
            "Higher values fine-tune more of the backbone.\n"
            "Grid: e.g. 0,5,10",
        )
        arch_form.addRow(self.row_unfreeze)

        self.row_dropout = _ParamRow(
            "Dropout (head)", "0.3",
            "Dropout probability applied to the prediction head during training.\n"
            "Lower values (0.2–0.3) reduce regularisation — helpful if the model underfits.\n"
            "Note: EfficientNet-B3 already has 0.3 internal dropout; MobileNet-V3 has 0.2.\n"
            "Stacking a large head dropout on top can cause underfitting for those architectures.\n"
            "Grid: e.g. 0.3,0.5",
        )
        arch_form.addRow(self.row_dropout)

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
        self.sp_epochs.setToolTip(
            "Maximum number of complete passes through the training set.\n"
            "Early stopping will usually halt training before this limit if the model\n"
            "stops improving, so it is safe to set this high (e.g. 100–200)."
        )
        ep_row.addWidget(self.sp_epochs)
        ep_row.addStretch()
        params_form.addRow("Epochs:", ep_row)

        self.row_bs = _ParamRow("Batch size", "32",
            "Number of images processed together in one forward/backward pass.\n"
            "Larger batches are faster but need more GPU memory.\n"
            "Typical values: 16 (low VRAM), 32 (default), 64–128 (high VRAM).\n"
            "Grid: separate multiple values with commas to search across them."
        )
        self.row_lr = _ParamRow("Learning rate", "0.001",
            "Controls how large each weight update step is.\n"
            "Too high → training diverges (loss explodes).\n"
            "Too low → very slow convergence.\n"
            "Good starting range: 0.001 (Adam/AdamW), 0.01 (SGD).\n"
            "Grid: try 0.01, 0.001, 0.0001 to find the sweet spot."
        )
        self.row_opt = _ParamRow("Optimizer", "Adam",
            "Algorithm used to update model weights after each batch.\n"
            "• Adam   – adaptive per-parameter LR; robust default for most tasks.\n"
            "• AdamW  – Adam with corrected weight decay; often slightly better.\n"
            "• SGD    – simple but needs careful LR and momentum tuning.\n"
            "• RMSprop – alternative adaptive method; less common.",
            choices=["Adam", "AdamW", "SGD", "RMSprop"],
        )
        self.row_wd = _ParamRow("Weight decay", "1e-4",
            "L2 regularisation: adds a penalty for large weight values.\n"
            "Reduces overfitting. Typical range: 0 (off) to 1e-3.\n"
            "Default 1e-4 (= 0.0001) works well for most cases."
        )
        self.row_loss = _ParamRow("Loss function", "MSE",
            "Metric minimised during training (regression mode only).\n"
            "• MSE     – mean squared error; penalises large errors heavily.\n"
            "• MAE     – mean absolute error; more robust to outliers.\n"
            "• Huber   – smooth blend of MSE (small errors) and MAE (large errors).\n"
            "• SmoothL1 – similar to Huber; often used in object detection.\n"
            "In classification mode CrossEntropyLoss is always used automatically.",
            choices=["MSE", "MAE", "Huber", "SmoothL1"],
        )
        for row in (self.row_bs, self.row_lr, self.row_opt, self.row_wd, self.row_loss):
            params_form.addRow(row)

        # Hide all Grid: controls until grid search is enabled
        for row in (self.row_arch, self.row_bs, self.row_lr, self.row_opt, self.row_wd, self.row_loss,
                    self.row_freeze, self.row_unfreeze, self.row_dropout):
            row.set_grid_visible(False)

        # SGD momentum (always single value)
        mom_row = QHBoxLayout()
        self.sp_momentum = QDoubleSpinBox()
        self.sp_momentum.setRange(0.0, 1.0)
        self.sp_momentum.setSingleStep(0.01)
        self.sp_momentum.setValue(0.9)
        self.sp_momentum.setDecimals(2)
        self.sp_momentum.setMaximumWidth(100)
        self.sp_momentum.setToolTip(
            "Only used when Optimizer = SGD.\n"
            "Controls how much the previous gradient direction influences the current step.\n"
            "0.9 is the standard default."
        )
        mom_row.addWidget(self.sp_momentum)
        mom_row.addWidget(QLabel("(SGD only)"))
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
        self.cmb_sched.setToolTip(
            "Controls how the learning rate changes over training:\n"
            "• None                – constant LR throughout.\n"
            "• StepLR              – multiply LR by Gamma every Step Size epochs.\n"
            "• CosineAnnealingLR   – smoothly decay LR from initial to Min LR over T_max\n"
            "                        epochs, then restart. Good for long runs.\n"
            "• ReduceLROnPlateau   – cut LR by Gamma when test loss stops improving.\n"
            "                        Recommended when you're not sure which schedule to use."
        )
        sched_form.addRow("Type:", self.cmb_sched)

        self.sp_step_size = QSpinBox()
        self.sp_step_size.setRange(1, 999)
        self.sp_step_size.setValue(10)
        self.sp_step_size.setToolTip(
            "StepLR only: reduce the learning rate every this many epochs.\n"
            "Example: step_size=10, gamma=0.5 → LR halves at epochs 10, 20, 30, …"
        )
        sched_form.addRow("Step size:", self.sp_step_size)

        self.sp_gamma = QDoubleSpinBox()
        self.sp_gamma.setRange(0.01, 1.0)
        self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setDecimals(2)
        self.sp_gamma.setValue(0.5)
        self.sp_gamma.setToolTip(
            "Multiplicative factor applied to LR when the scheduler fires.\n"
            "• StepLR: new_LR = LR × gamma every step_size epochs.\n"
            "• ReduceLROnPlateau: new_LR = LR × gamma when progress stalls.\n"
            "Values < 1 decrease LR; 0.5 halves it, 0.1 reduces by 10×."
        )
        sched_form.addRow("Gamma:", self.sp_gamma)

        self.sp_tmax = QSpinBox()
        self.sp_tmax.setRange(1, 9999)
        self.sp_tmax.setValue(50)
        self.sp_tmax.setToolTip(
            "CosineAnnealingLR only: half-period of the cosine cycle in epochs.\n"
            "LR decays from the initial value to Min LR over T_max epochs, then restarts.\n"
            "Set to your expected total training length for a single decay cycle."
        )
        sched_form.addRow("T_max (cosine):", self.sp_tmax)

        self.sp_plat_patience = QSpinBox()
        self.sp_plat_patience.setRange(1, 999)
        self.sp_plat_patience.setValue(5)
        self.sp_plat_patience.setToolTip(
            "ReduceLROnPlateau only: epochs with no improvement in test loss before LR is cut.\n"
            "Smaller values reduce LR quickly; larger values give more time to escape plateaus."
        )
        sched_form.addRow("Plateau patience:", self.sp_plat_patience)

        self.sp_min_lr = QDoubleSpinBox()
        self.sp_min_lr.setRange(0.0, 0.1)
        self.sp_min_lr.setSingleStep(1e-6)
        self.sp_min_lr.setDecimals(8)
        self.sp_min_lr.setValue(1e-6)
        self.sp_min_lr.setToolTip(
            "Floor for the learning rate — the scheduler will never reduce it below this value.\n"
            "Prevents LR from becoming so small that training effectively stops."
        )
        sched_form.addRow("Min LR:", self.sp_min_lr)

        # Early stopping
        es_grp = QGroupBox("Early Stopping")
        cfg_v.addWidget(es_grp)
        es_form = QFormLayout(es_grp)
        es_form.setLabelAlignment(Qt.AlignRight)

        self.chk_early_stop = QCheckBox("Enable")
        self.chk_early_stop.setChecked(True)
        self.chk_early_stop.setToolTip(
            "Stop training automatically when the model stops improving.\n"
            "Prevents wasting time and reduces overfitting. Strongly recommended."
        )
        es_form.addRow("", self.chk_early_stop)

        self.sp_es_patience = QSpinBox()
        self.sp_es_patience.setRange(1, 999)
        self.sp_es_patience.setValue(15)
        self.sp_es_patience.setToolTip(
            "Consecutive epochs with no improvement before training halts.\n"
            "Larger values give the model more chances to escape temporary plateaus.\n"
            "Typical range: 5–20. Increase if using a LR scheduler with slow reductions."
        )
        es_form.addRow("Patience (epochs):", self.sp_es_patience)

        self.sp_es_delta = QDoubleSpinBox()
        self.sp_es_delta.setRange(0.0, 1.0)
        self.sp_es_delta.setSingleStep(1e-4)
        self.sp_es_delta.setDecimals(6)
        self.sp_es_delta.setValue(1e-4)
        self.sp_es_delta.setToolTip(
            "Minimum change that counts as an improvement.\n"
            "The patience counter only resets when improvement exceeds this threshold.\n"
            "1e-4 (= 0.0001) is a safe default; set to 0 to count any improvement."
        )
        es_form.addRow("Min delta:", self.sp_es_delta)

        # Output
        out_grp = QGroupBox("Output")
        cfg_v.addWidget(out_grp)
        out_form = QFormLayout(out_grp)
        out_form.setLabelAlignment(Qt.AlignRight)

        out_row = QHBoxLayout()
        self.edit_out_dir = QLineEdit("./results")
        self.edit_out_dir.setToolTip(
            "Folder where trained model checkpoints, label CSV, and result JSON files are saved.\n"
            "Each run creates a sub-folder named by its run ID (e.g. run_0001).\n"
            "The Results tab loads from this directory."
        )
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.edit_out_dir)
        out_row.addWidget(btn_browse_out)
        out_form.addRow("Output directory:", out_row)

        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(0, 32)
        self.sp_workers.setValue(0)
        self.sp_workers.setToolTip(
            "Number of parallel CPU processes used to load and augment images.\n"
            "0 = load in the main process (required on Windows; always safe).\n"
            "On Linux/Mac, 2–4 workers can speed up GPU-bound training\n"
            "by preparing the next batch while the GPU works on the current one."
        )
        out_form.addRow("DataLoader workers:", self.sp_workers)

        self.chk_amp = QCheckBox("Automatic Mixed Precision (AMP)")
        self.chk_amp.setToolTip(
            "Train using 16-bit floating point where possible (requires Nvidia GPU with Tensor Cores).\n"
            "Can speed up training by 1.5–3× and halve GPU memory usage with minimal accuracy loss.\n"
            "Safe to enable for GTX 10xx and newer GPUs; has no effect when training on CPU."
        )
        out_form.addRow("", self.chk_amp)

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

        # Live metrics  (Val = validation split, monitored each epoch)
        metrics_grp = QGroupBox("Current Metrics  (Validation split — monitored each epoch)")
        metrics_h = QHBoxLayout(metrics_grp)
        self.lbl_train_loss = _metric_label("Train Loss", "—")
        self.lbl_val_loss   = _metric_label("Val Loss",   "—")
        self.lbl_val_rmse   = _metric_label("Val RMSE",   "—")
        self.lbl_val_mae    = _metric_label("Val MAE",    "—")
        self.lbl_val_acc    = _metric_label("Val Acc",    "—")
        self.lbl_val_f1     = _metric_label("Val F1",     "—")
        for w in (self.lbl_train_loss, self.lbl_val_loss,
                  self.lbl_val_rmse, self.lbl_val_mae,
                  self.lbl_val_acc, self.lbl_val_f1):
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
        for row in (self.row_arch, self.row_bs, self.row_lr, self.row_opt, self.row_wd, self.row_loss,
                    self.row_freeze, self.row_unfreeze, self.row_dropout):
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

        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if use_grid:
            worker = GridSearchWorker(full_config)
            worker.run_log.connect(self._on_log)
            worker.run_started.connect(self._on_gs_run_started)
            worker.run_progress.connect(self._on_gs_progress)
            worker.run_finished.connect(self._on_gs_run_finished)
            worker.epoch_metrics.connect(self._on_epoch_metrics)
            worker.all_done.connect(self._on_all_done)
            worker.error.connect(self._on_error)
            self.lbl_status.setText("Grid search running…")
        else:
            worker = TrainingWorker(full_config, run_id=f"run_{ts}")
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

    @staticmethod
    def _set_metric(widget, text: str):
        """Null-safe helper: find the 'value' child label and set its text."""
        lbl = widget.findChild(QLabel, "value")
        if lbl is not None:
            lbl.setText(text)

    def _on_epoch_metrics(self, m: dict):
        self._set_metric(self.lbl_train_loss, f"{m.get('train_loss', 0):.4f}")
        vl = m.get("val_loss")
        self._set_metric(self.lbl_val_loss,  f"{vl:.4f}" if vl is not None else "—")
        vr = m.get("val_rmse")
        self._set_metric(self.lbl_val_rmse,  f"{vr:.4f}" if vr is not None else "—")
        vm = m.get("val_mae")
        self._set_metric(self.lbl_val_mae,   f"{vm:.4f}" if vm is not None else "—")
        va = m.get("val_accuracy")
        self._set_metric(self.lbl_val_acc,   f"{va:.3f}" if va is not None else "—")
        vf = m.get("val_f1")
        self._set_metric(self.lbl_val_f1,    f"{vf:.3f}" if vf is not None else "—")

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
        # Add this run's result to the panel immediately — don't wait for all
        # runs to finish.  This way the user sees each result as it completes,
        # and a crash mid-grid-search doesn't lose the already-completed rows.
        mw = self.window()
        if hasattr(mw, "res_panel"):
            mw.res_panel.set_output_dir(self.edit_out_dir.text())
            mw.res_panel.add_result(result)

    def _on_single_done(self, result: dict):
        self.pbar_runs.setValue(1)
        self._training_finished([result])

    def _on_all_done(self, results: list):
        # Individual results were already added to the panel one by one in
        # _on_gs_run_finished, so we only need to finalize the UI here.
        self.lbl_status.setText(f"Grid search done  ({len(results)} run(s))")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        mw = self.window()
        if hasattr(mw, "tabs") and hasattr(mw, "res_panel"):
            mw.tabs.setCurrentWidget(mw.res_panel)
        if hasattr(mw, "lbl_status"):
            mw.lbl_status.setText(f"Grid search complete — {len(results)} run(s)")

    def _on_error(self, msg: str):
        self.log_edit.append(f"\n[ERROR]\n{msg}")
        self.lbl_status.setText("Error — see log")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        try:
            from ..utils.crash_reporter import log_worker_crash
            worker = getattr(self, "_worker", None) or getattr(self, "_gs_worker", None)
            run_id = getattr(worker, "run_id", "unknown") if worker else "unknown"
            log_worker_crash(run_id, msg)
        except Exception:
            pass

    def _training_finished(self, results: list):
        self.lbl_status.setText(f"Done  ({len(results)} run(s))")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Emit results to main window for results panel
        mw = self.window()
        if hasattr(mw, "receive_training_results"):
            mw.receive_training_results(results)

    def _reset_metrics(self):
        for w in (self.lbl_train_loss, self.lbl_val_loss,
                  self.lbl_val_rmse, self.lbl_val_mae,
                  self.lbl_val_acc, self.lbl_val_f1):
            vl = w.findChild(QLabel, "value")
            if vl:
                vl.setText("—")

    # ---------------------------------------------------------------- public API
    def get_config(self) -> dict:
        return {
            # Architecture / transfer learning settings
            "architecture":     self.row_arch.get_entry(),
            "pretrained":       self.chk_pretrained.isChecked(),
            "freeze_backbone":  self.row_freeze.get_entry(),
            "unfreeze_last_n":  self.row_unfreeze.get_entry(),
            "dropout_head":     self.row_dropout.get_entry(),
            # Hyperparameters
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
            "use_amp": self.chk_amp.isChecked(),
        }

    def set_config(self, cfg: dict):
        # Architecture / transfer learning
        self.row_arch.set_entry(cfg.get("architecture", "ResNet-50"))
        self.chk_pretrained.setChecked(bool(cfg.get("pretrained", True)))
        # freeze/unfreeze/dropout accept both old scalar values and new dict format
        self.row_freeze.set_entry(cfg.get("freeze_backbone", "True"))
        self.row_unfreeze.set_entry(cfg.get("unfreeze_last_n", "0"))
        self.row_dropout.set_entry(cfg.get("dropout_head", "0.3"))
        # Hyperparameters
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
        self.chk_amp.setChecked(bool(cfg.get("use_amp", False)))


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
