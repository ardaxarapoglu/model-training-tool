"""Model configuration panel: transfer learning vs from-scratch CNN."""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QStackedWidget, QScrollArea,
    QFrame, QLineEdit, QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from ..core.model_builder import ARCHITECTURES


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

        # ---- Mode selector ----
        grp_mode = QGroupBox("Training Mode")
        cv.addWidget(grp_mode)
        mode_v = QVBoxLayout(grp_mode)

        mode_row = QHBoxLayout()
        self.rb_transfer = QRadioButton("Transfer Learning  (recommended)")
        self.rb_scratch  = QRadioButton("Train from Scratch")
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
        tp = QFormLayout(transfer_page)
        tp.setLabelAlignment(Qt.AlignRight)
        tp.setHorizontalSpacing(12)

        self.cmb_arch = QComboBox()
        for name in ARCHITECTURES:
            self.cmb_arch.addItem(name, name)
        self.cmb_arch.setCurrentText("ResNet-50")
        tp.addRow("Architecture:", self.cmb_arch)

        arch_info = QLabel(
            "ResNet / VGG / DenseNet / MobileNet / EfficientNet families available."
        )
        arch_info.setStyleSheet("color:#666;font-size:11px;")
        tp.addRow("", arch_info)

        self.chk_pretrained = QCheckBox("Use pretrained ImageNet weights")
        self.chk_pretrained.setChecked(True)
        tp.addRow("", self.chk_pretrained)

        self.chk_freeze = QCheckBox("Freeze backbone  (only train head)")
        tp.addRow("", self.chk_freeze)

        unfreeze_row = QHBoxLayout()
        self.sp_unfreeze = QSpinBox()
        self.sp_unfreeze.setRange(0, 500)
        self.sp_unfreeze.setValue(0)
        self.sp_unfreeze.setToolTip(
            "Number of parameter tensors from the end of the backbone to unfreeze. "
            "0 = respect freeze-backbone setting exactly."
        )
        unfreeze_row.addWidget(self.sp_unfreeze)
        unfreeze_row.addWidget(QLabel("last param tensors (0 = off)"))
        unfreeze_row.addStretch()
        tp.addRow("Unfreeze last N:", unfreeze_row)

        self.sp_dropout_t = QDoubleSpinBox()
        self.sp_dropout_t.setRange(0.0, 0.9)
        self.sp_dropout_t.setSingleStep(0.05)
        self.sp_dropout_t.setDecimals(2)
        self.sp_dropout_t.setValue(0.5)
        tp.addRow("Dropout (head):", self.sp_dropout_t)

        # Grid search for architecture
        tp.addRow(QLabel(""))
        gs_lbl = QLabel("Architecture grid search  (comma-separated, only active in grid-search mode):")
        gs_lbl.setStyleSheet("color:#555;font-size:11px;")
        tp.addRow("", gs_lbl)
        self.chk_arch_grid = QCheckBox("Include architecture in grid search")
        tp.addRow("", self.chk_arch_grid)
        self.edit_arch_values = QLineEdit()
        self.edit_arch_values.setPlaceholderText("e.g., ResNet-50,ResNet-101,EfficientNet-B0")
        self.edit_arch_values.setEnabled(False)
        self.chk_arch_grid.toggled.connect(self.edit_arch_values.setEnabled)
        tp.addRow("Architectures:", self.edit_arch_values)

        self.stack.addWidget(transfer_page)

        # -- Scratch page --
        scratch_page = QWidget()
        sp_layout = QFormLayout(scratch_page)
        sp_layout.setLabelAlignment(Qt.AlignRight)
        sp_layout.setHorizontalSpacing(12)

        self.sp_conv_blocks = QSpinBox()
        self.sp_conv_blocks.setRange(2, 8)
        self.sp_conv_blocks.setValue(4)
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
        sp_layout.addRow("", self.chk_batch_norm)

        self.sp_dropout_s = QDoubleSpinBox()
        self.sp_dropout_s.setRange(0.0, 0.9)
        self.sp_dropout_s.setSingleStep(0.05)
        self.sp_dropout_s.setDecimals(2)
        self.sp_dropout_s.setValue(0.5)
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

    # ---------------------------------------------------------------- public API
    def get_config(self) -> dict:
        mode = "transfer" if self.rb_transfer.isChecked() else "scratch"
        arch_vals = [v.strip() for v in self.edit_arch_values.text().split(",") if v.strip()]
        return {
            "mode": mode,
            "transfer": {
                "architecture": self.cmb_arch.currentText(),
                "pretrained": self.chk_pretrained.isChecked(),
                "freeze_backbone": self.chk_freeze.isChecked(),
                "unfreeze_last_n": self.sp_unfreeze.value(),
                "dropout": self.sp_dropout_t.value(),
                "architecture_grid": {
                    "use_grid": self.chk_arch_grid.isChecked(),
                    "values": self.edit_arch_values.text(),
                    "value": self.cmb_arch.currentText(),
                },
            },
            "scratch": {
                "num_conv_blocks": self.sp_conv_blocks.value(),
                "base_filters": self.sp_base_filters.value(),
                "fc_layers": _parse_ints(self.edit_fc_layers.text(), [256, 128]),
                "batch_norm": self.chk_batch_norm.isChecked(),
                "dropout": self.sp_dropout_s.value(),
            },
        }

    def set_config(self, cfg: dict):
        mode = cfg.get("mode", "transfer")
        if mode == "transfer":
            self.rb_transfer.setChecked(True)
        else:
            self.rb_scratch.setChecked(True)

        tr = cfg.get("transfer", {})
        arch = tr.get("architecture", "ResNet-50")
        idx = self.cmb_arch.findText(arch)
        if idx >= 0:
            self.cmb_arch.setCurrentIndex(idx)
        self.chk_pretrained.setChecked(tr.get("pretrained", True))
        self.chk_freeze.setChecked(tr.get("freeze_backbone", False))
        self.sp_unfreeze.setValue(int(tr.get("unfreeze_last_n", 0)))
        self.sp_dropout_t.setValue(float(tr.get("dropout", 0.5)))

        arch_grid = tr.get("architecture_grid", {})
        self.chk_arch_grid.setChecked(arch_grid.get("use_grid", False))
        self.edit_arch_values.setText(arch_grid.get("values", ""))

        sc = cfg.get("scratch", {})
        self.sp_conv_blocks.setValue(int(sc.get("num_conv_blocks", 4)))
        self.sp_base_filters.setValue(int(sc.get("base_filters", 32)))
        self.edit_fc_layers.setText(", ".join(str(v) for v in sc.get("fc_layers", [256, 128])))
        self.chk_batch_norm.setChecked(sc.get("batch_norm", True))
        self.sp_dropout_s.setValue(float(sc.get("dropout", 0.5)))


def _parse_ints(text, default):
    try:
        vals = [int(v.strip()) for v in text.split(",") if v.strip()]
        return vals if vals else default
    except Exception:
        return default
