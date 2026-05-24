"""Model configuration panel: transfer learning vs from-scratch CNN."""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QStackedWidget, QScrollArea,
    QFrame, QLineEdit, QSizePolicy, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QColor


class ModelPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        cv = QVBoxLayout(container)
        cv.setSpacing(10)

        # ---- Classification / Regression selector ----
        grp_out = QGroupBox("Output Mode")
        cv.addWidget(grp_out)
        out_v = QVBoxLayout(grp_out)

        out_mode_row = QHBoxLayout()
        self.rb_regression = QRadioButton("Regression  (predict Pb% as a continuous value)")
        self.rb_regression.setToolTip(
            "The model outputs a single number representing the predicted Pb concentration.\n"
            "Evaluated with RMSE, MAE, and R² metrics.\n"
            "Best when you need a precise numeric estimate."
        )
        self.rb_classify   = QRadioButton("Classification  (predict a named category)")
        self.rb_classify.setToolTip(
            "The model assigns each image to one of the named classes you define below.\n"
            "Evaluated with accuracy, F1 score, and confusion matrix.\n"
            "Best when you need a simple actionable label (e.g. Good / Acceptable / Bad)."
        )
        self.rb_regression.setChecked(True)
        bg_out = QButtonGroup(self)
        bg_out.addButton(self.rb_regression)
        bg_out.addButton(self.rb_classify)
        out_mode_row.addWidget(self.rb_regression)
        out_mode_row.addWidget(self.rb_classify)
        out_mode_row.addStretch()
        out_v.addLayout(out_mode_row)

        self.cls_editor = QWidget()
        ce_v = QVBoxLayout(self.cls_editor)
        ce_v.setContentsMargins(0, 4, 0, 0)
        ce_v.setSpacing(4)

        info_lbl = QLabel(
            "Define classes in order of increasing Pb%.  "
            "The last class has no upper bound and catches all remaining values."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color:#555;font-size:11px;")
        ce_v.addWidget(info_lbl)

        self.tbl_classes = QTableWidget(0, 2)
        self.tbl_classes.setHorizontalHeaderLabels(["Class Name", "Upper Bound (%)"])
        self.tbl_classes.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_classes.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_classes.setMinimumHeight(280)
        self.tbl_classes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.tbl_classes.verticalHeader().setVisible(False)
        ce_v.addWidget(self.tbl_classes)

        cls_btn_row = QHBoxLayout()
        btn_add_cls = QPushButton("+ Add Class")
        btn_add_cls.clicked.connect(self._add_class_row)
        btn_rm_cls  = QPushButton("- Remove Last")
        btn_rm_cls.clicked.connect(self._remove_class_row)
        cls_btn_row.addWidget(btn_add_cls)
        cls_btn_row.addWidget(btn_rm_cls)
        cls_btn_row.addStretch()
        ce_v.addLayout(cls_btn_row)

        out_v.addWidget(self.cls_editor)
        self.cls_editor.setVisible(False)
        self.rb_classify.toggled.connect(self.cls_editor.setVisible)

        self._init_default_classes()

        # ---- Training Mode selector ----
        grp_mode = QGroupBox("Training Mode")
        cv.addWidget(grp_mode)
        mode_v = QVBoxLayout(grp_mode)

        mode_row = QHBoxLayout()
        self.rb_transfer = QRadioButton("Transfer Learning  (recommended)")
        self.rb_transfer.setToolTip(
            "Start from a model pre-trained on ImageNet (1.2 million photos).\n"
            "The network already knows how to detect edges, textures, and shapes.\n"
            "Only the final prediction head is re-trained (or the whole network fine-tuned).\n"
            "Reaches good results with far fewer froth images and training epochs."
        )
        self.rb_scratch  = QRadioButton("Train from Scratch")
        self.rb_scratch.setToolTip(
            "Build and train a simple CNN using only your froth images.\n"
            "Requires more data and epochs to reach similar accuracy.\n"
            "Useful if the froth texture is very different from natural photos."
        )
        self.rb_transfer.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_transfer)
        bg.addButton(self.rb_scratch)
        mode_row.addWidget(self.rb_transfer)
        mode_row.addWidget(self.rb_scratch)
        mode_row.addStretch()
        mode_v.addLayout(mode_row)

        self.rb_transfer.toggled.connect(self._on_mode_changed)

        # ---- Stacked widget: transfer / scratch ----
        self.stack = QStackedWidget()
        cv.addWidget(self.stack)

        # -- Transfer page --
        transfer_page = QWidget()
        tp_v = QVBoxLayout(transfer_page)
        tp_v.setContentsMargins(8, 8, 8, 8)
        tp_info = QLabel(
            "Architecture selection, pretrained weights, freeze/unfreeze, and dropout settings\n"
            "have been moved to the  <b>④ Training</b>  tab so they can be included\n"
            "in grid search alongside hyperparameters."
        )
        tp_info.setWordWrap(True)
        tp_info.setStyleSheet("color:#555; font-size:11px; margin-top:8px;")
        tp_v.addWidget(tp_info)
        tp_v.addStretch()
        self.stack.addWidget(transfer_page)

        # -- Scratch page --
        scratch_page = QWidget()
        sp_layout = QFormLayout(scratch_page)
        sp_layout.setLabelAlignment(Qt.AlignRight)
        sp_layout.setHorizontalSpacing(12)

        self.sp_conv_blocks = QSpinBox()
        self.sp_conv_blocks.setRange(2, 8)
        self.sp_conv_blocks.setValue(4)
        self.sp_conv_blocks.setToolTip(
            "Number of convolutional blocks stacked in the network.\n"
            "Each block doubles the number of feature channels.\n"
            "More blocks = more capacity but slower training and more risk of overfitting.\n"
            "4 blocks is a reasonable starting point for 224×224 images."
        )
        sp_layout.addRow("Conv blocks:", self.sp_conv_blocks)

        self.sp_base_filters = QSpinBox()
        self.sp_base_filters.setRange(8, 256)
        self.sp_base_filters.setValue(32)
        self.sp_base_filters.setToolTip(
            "Filters in first conv block. Each block doubles the filters."
        )
        sp_layout.addRow("Base filters:", self.sp_base_filters)

        self.edit_fc_layers = QLineEdit("256, 128")
        self.edit_fc_layers.setToolTip("Comma-separated list of FC layer sizes after global pooling.")
        sp_layout.addRow("FC layer sizes:", self.edit_fc_layers)

        self.chk_batch_norm = QCheckBox("Batch normalization")
        self.chk_batch_norm.setChecked(True)
        self.chk_batch_norm.setToolTip(
            "Normalise the output of each conv block so it has mean≈0 and std≈1.\n"
            "Stabilises training, allows higher learning rates, and usually improves accuracy.\n"
            "Almost always beneficial — only disable if you're experimenting."
        )
        sp_layout.addRow("", self.chk_batch_norm)

        self.sp_dropout_s = QDoubleSpinBox()
        self.sp_dropout_s.setRange(0.0, 0.9)
        self.sp_dropout_s.setSingleStep(0.05)
        self.sp_dropout_s.setDecimals(2)
        self.sp_dropout_s.setValue(0.5)
        self.sp_dropout_s.setToolTip(
            "Randomly zeros this fraction of neurons in the FC layers during training.\n"
            "Reduces overfitting. 0.5 is a strong default; reduce to 0.2–0.3 if underfitting."
        )
        sp_layout.addRow("Dropout:", self.sp_dropout_s)

        sp_info = QLabel(
            "Architecture: Conv(3×3) → BN → ReLU → MaxPool  ×  N blocks → "
            "AdaptiveAvgPool(4×4) → Flatten → FC layers → Output(1)"
        )
        sp_info.setWordWrap(True)
        sp_info.setStyleSheet("color:#666;font-size:11px;margin-top:6px;")
        sp_layout.addRow("", sp_info)

        self.stack.addWidget(scratch_page)

        self._on_mode_changed(True)
        cv.addStretch()

    def _on_mode_changed(self, transfer: bool):
        self.stack.setCurrentIndex(0 if transfer else 1)

    # ---------------------------------------------------------------- class editor helpers
    def _init_default_classes(self):
        self._set_classes([
            {"name": "Bad",        "max": 20.0},
            {"name": "Acceptable", "max": 40.0},
            {"name": "Good",       "max": None},
        ])

    def _add_class_row(self):
        n = self.tbl_classes.rowCount()
        # Insert before last "∞" row
        self.tbl_classes.insertRow(n - 1)
        self.tbl_classes.setItem(n - 1, 0, QTableWidgetItem(f"Class {n}"))
        self.tbl_classes.setItem(n - 1, 1, QTableWidgetItem("50"))

    def _remove_class_row(self):
        n = self.tbl_classes.rowCount()
        if n <= 2:
            return
        self.tbl_classes.removeRow(n - 2)

    def _get_classes(self) -> list:
        classes = []
        for row in range(self.tbl_classes.rowCount()):
            name_item  = self.tbl_classes.item(row, 0)
            bound_item = self.tbl_classes.item(row, 1)
            name = name_item.text().strip() if name_item else f"Class {row + 1}"
            bound_text = (bound_item.text() if bound_item else "∞").strip()
            try:
                max_val = None if bound_text in ("∞", "", "inf") else float(bound_text)
            except ValueError:
                max_val = None
            classes.append({"name": name, "max": max_val})
        return classes

    def _set_classes(self, classes: list):
        self.tbl_classes.setRowCount(len(classes))
        for row, cls in enumerate(classes):
            name    = cls.get("name", f"Class {row + 1}")
            max_val = cls.get("max")
            bound_text = "∞" if max_val is None else str(max_val)
            self.tbl_classes.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(bound_text)
            if max_val is None:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setForeground(QColor("#888888"))
            self.tbl_classes.setItem(row, 1, item)

    # ---------------------------------------------------------------- public API
    def get_config(self) -> dict:
        return {
            "mode": "transfer" if self.rb_transfer.isChecked() else "scratch",
            "transfer": {},   # architecture/pretrained/freeze/dropout now in training config
            "scratch": {
                "num_conv_blocks": self.sp_conv_blocks.value(),
                "base_filters": self.sp_base_filters.value(),
                "fc_layers": _parse_ints(self.edit_fc_layers.text(), [256, 128]),
                "batch_norm": self.chk_batch_norm.isChecked(),
                "dropout": self.sp_dropout_s.value(),
            },
            "classification": {
                "enabled": self.rb_classify.isChecked(),
                "classes": self._get_classes(),
            },
        }

    def set_config(self, cfg: dict):
        mode = cfg.get("mode", "transfer")
        if mode == "transfer":
            self.rb_transfer.setChecked(True)
        else:
            self.rb_scratch.setChecked(True)

        sc = cfg.get("scratch", {})
        self.sp_conv_blocks.setValue(int(sc.get("num_conv_blocks", 4)))
        self.sp_base_filters.setValue(int(sc.get("base_filters", 32)))
        self.edit_fc_layers.setText(", ".join(str(v) for v in sc.get("fc_layers", [256, 128])))
        self.chk_batch_norm.setChecked(sc.get("batch_norm", True))
        self.sp_dropout_s.setValue(float(sc.get("dropout", 0.5)))

        cls_cfg = cfg.get("classification", {})
        is_cls = cls_cfg.get("enabled", False)
        self.rb_classify.setChecked(is_cls)
        self.rb_regression.setChecked(not is_cls)
        if cls_cfg.get("classes"):
            self._set_classes(cls_cfg["classes"])


def _parse_ints(text, default):
    try:
        vals = [int(v.strip()) for v in text.split(",") if v.strip()]
        return vals if vals else default
    except Exception:
        return default
