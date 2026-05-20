"""Preprocessing panel: crop, resize, normalization, and augmentation settings."""
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QScrollArea, QFrame, QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


class _DSpinRow(QWidget):
    """Label + QDoubleSpinBox in a single row widget."""
    def __init__(self, label, default, lo=0.0, hi=1.0, step=0.05, decimals=2, suffix=""):
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(QLabel(label))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default)
        self.spin.setSuffix(suffix)
        hl.addWidget(self.spin)


class AugmentRow(QWidget):
    """Checkbox to enable an augmentation + optional parameter widgets."""
    def __init__(self, title: str, params: list = None):
        """params: list of (label, default, lo, hi, step, decimals) tuples."""
        super().__init__()
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 2, 0, 2)
        self.chk = QCheckBox(title)
        self.chk.setMinimumWidth(200)
        hl.addWidget(self.chk)
        self.spins = []
        for label, default, lo, hi, step, decimals in (params or []):
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#555;margin-left:8px;")
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(default)
            sp.setMaximumWidth(90)
            hl.addWidget(lbl)
            hl.addWidget(sp)
            self.spins.append(sp)
        hl.addStretch()
        self.chk.toggled.connect(self._on_toggle)
        self._on_toggle(False)

    def _on_toggle(self, checked):
        for sp in self.spins:
            sp.setEnabled(checked)

    def is_enabled(self) -> bool:
        return self.chk.isChecked()

    def values(self) -> list:
        return [sp.value() for sp in self.spins]


class PreprocessingPanel(QWidget):
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

        # ---- Crop ----
        grp_crop = QGroupBox("Cropping")
        cv.addWidget(grp_crop)
        grp_v = QVBoxLayout(grp_crop)

        self.chk_crop = QCheckBox("Enable cropping")
        grp_v.addWidget(self.chk_crop)

        mode_row = QHBoxLayout()
        self.rb_manual = QRadioButton("Manual (x, y, w, h)")
        self.rb_center = QRadioButton("Center crop (size)")
        self.rb_manual.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_manual)
        bg.addButton(self.rb_center)
        mode_row.addWidget(self.rb_manual)
        mode_row.addWidget(self.rb_center)
        mode_row.addStretch()
        grp_v.addLayout(mode_row)

        manual_row = QHBoxLayout()
        for lbl, attr in [("x:", "sp_cx"), ("y:", "sp_cy"), ("w:", "sp_cw"), ("h:", "sp_ch")]:
            manual_row.addWidget(QLabel(lbl))
            sp = QSpinBox()
            sp.setRange(0, 9999)
            sp.setValue(0 if lbl in ("x:", "y:") else 224)
            sp.setMaximumWidth(80)
            setattr(self, attr, sp)
            manual_row.addWidget(sp)
        manual_row.addStretch()
        grp_v.addLayout(manual_row)

        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Size:"))
        self.sp_center_size = QSpinBox()
        self.sp_center_size.setRange(16, 9999)
        self.sp_center_size.setValue(224)
        center_row.addWidget(self.sp_center_size)
        center_row.addStretch()
        grp_v.addLayout(center_row)

        self.rb_manual.toggled.connect(self._update_crop_mode)
        self.chk_crop.toggled.connect(self._update_crop_mode)
        self._update_crop_mode()

        # ---- Resize ----
        grp_resize = QGroupBox("Resize (final input to model)")
        cv.addWidget(grp_resize)
        resize_form = QFormLayout(grp_resize)
        resize_form.setLabelAlignment(Qt.AlignRight)

        rw = QHBoxLayout()
        self.sp_rw = QSpinBox()
        self.sp_rw.setRange(16, 1024)
        self.sp_rw.setValue(224)
        self.sp_rh = QSpinBox()
        self.sp_rh.setRange(16, 1024)
        self.sp_rh.setValue(224)
        rw.addWidget(self.sp_rw)
        rw.addWidget(QLabel("×"))
        rw.addWidget(self.sp_rh)
        rw.addStretch()
        resize_form.addRow("Width × Height (px):", rw)

        # ---- Normalization ----
        grp_norm = QGroupBox("Normalization")
        cv.addWidget(grp_norm)
        norm_v = QVBoxLayout(grp_norm)
        self.chk_imagenet_norm = QCheckBox("Use ImageNet mean/std  (recommended for transfer learning)")
        self.chk_imagenet_norm.setChecked(True)
        norm_v.addWidget(self.chk_imagenet_norm)

        custom_row = QFormLayout()
        custom_row.setLabelAlignment(Qt.AlignRight)
        self.edit_mean = _make_line("0.485, 0.456, 0.406")
        self.edit_std = _make_line("0.229, 0.224, 0.225")
        custom_row.addRow("Mean (R,G,B):", self.edit_mean)
        custom_row.addRow("Std  (R,G,B):", self.edit_std)
        norm_v.addLayout(custom_row)
        self.chk_imagenet_norm.toggled.connect(
            lambda checked: (self.edit_mean.setEnabled(not checked), self.edit_std.setEnabled(not checked))
        )
        self.chk_imagenet_norm.toggled.emit(True)

        # ---- Augmentation ----
        grp_aug = QGroupBox("Data Augmentation  (applied to training set only)")
        cv.addWidget(grp_aug)
        aug_v = QVBoxLayout(grp_aug)

        aug_v.addWidget(_section_label("Flips & Rotation"))
        self.aug_hflip = AugmentRow("Horizontal Flip", [("p:", 0.5, 0.0, 1.0, 0.05, 2)])
        self.aug_vflip = AugmentRow("Vertical Flip",   [("p:", 0.5, 0.0, 1.0, 0.05, 2)])
        self.aug_rot   = AugmentRow("Random Rotation",  [("degrees:", 15, 0, 180, 1, 0)])
        self.aug_hflip.chk.setChecked(True)
        aug_v.addWidget(self.aug_hflip)
        aug_v.addWidget(self.aug_vflip)
        aug_v.addWidget(self.aug_rot)

        aug_v.addWidget(_section_label("Color"))
        self.aug_cj = AugmentRow("Color Jitter", [
            ("brightness:", 0.2, 0.0, 2.0, 0.05, 2),
            ("contrast:",   0.2, 0.0, 2.0, 0.05, 2),
            ("saturation:", 0.2, 0.0, 2.0, 0.05, 2),
            ("hue:",        0.05, 0.0, 0.5, 0.01, 2),
        ])
        self.aug_cj.chk.setChecked(True)
        aug_v.addWidget(self.aug_cj)

        aug_v.addWidget(_section_label("Spatial"))
        self.aug_rrc = AugmentRow("Random Resized Crop", [
            ("scale min:", 0.8, 0.1, 1.0, 0.05, 2),
            ("scale max:", 1.0, 0.1, 1.0, 0.05, 2),
        ])
        aug_v.addWidget(self.aug_rrc)

        aug_v.addWidget(_section_label("Noise / Regularization"))
        self.aug_blur = AugmentRow("Gaussian Blur", [("kernel:", 5, 3, 21, 2, 0)])
        self.aug_erase = AugmentRow("Random Erasing", [("p:", 0.2, 0.0, 1.0, 0.05, 2)])
        aug_v.addWidget(self.aug_blur)
        aug_v.addWidget(self.aug_erase)

        cv.addStretch()

    # ---------------------------------------------------------------- helpers
    def _update_crop_mode(self):
        enabled = self.chk_crop.isChecked()
        manual = self.rb_manual.isChecked()
        for sp in (self.sp_cx, self.sp_cy, self.sp_cw, self.sp_ch):
            sp.setEnabled(enabled and manual)
        self.sp_center_size.setEnabled(enabled and not manual)
        self.rb_manual.setEnabled(enabled)
        self.rb_center.setEnabled(enabled)

    # ---------------------------------------------------------------- public API
    def get_config(self) -> dict:
        crop_type = "manual" if self.rb_manual.isChecked() else "center"
        mean = _parse_floats(self.edit_mean.text(), [0.485, 0.456, 0.406])
        std  = _parse_floats(self.edit_std.text(),  [0.229, 0.224, 0.225])
        return {
            "crop": {
                "enabled": self.chk_crop.isChecked(),
                "type": crop_type,
                "x": self.sp_cx.value(),
                "y": self.sp_cy.value(),
                "w": self.sp_cw.value(),
                "h": self.sp_ch.value(),
                "center_size": self.sp_center_size.value(),
            },
            "resize": {"width": self.sp_rw.value(), "height": self.sp_rh.value()},
            "normalize": {
                "use_imagenet": self.chk_imagenet_norm.isChecked(),
                "mean": mean,
                "std": std,
            },
            "augmentation": {
                "h_flip": {"enabled": self.aug_hflip.is_enabled(), "p": self.aug_hflip.values()[0]},
                "v_flip": {"enabled": self.aug_vflip.is_enabled(), "p": self.aug_vflip.values()[0]},
                "rotation": {"enabled": self.aug_rot.is_enabled(), "degrees": self.aug_rot.values()[0]},
                "color_jitter": {
                    "enabled": self.aug_cj.is_enabled(),
                    "brightness": self.aug_cj.values()[0],
                    "contrast":   self.aug_cj.values()[1],
                    "saturation": self.aug_cj.values()[2],
                    "hue":        self.aug_cj.values()[3],
                },
                "random_resized_crop": {
                    "enabled":   self.aug_rrc.is_enabled(),
                    "scale_min": self.aug_rrc.values()[0],
                    "scale_max": self.aug_rrc.values()[1],
                },
                "gaussian_blur": {"enabled": self.aug_blur.is_enabled(), "kernel_size": int(self.aug_blur.values()[0])},
                "random_erasing": {"enabled": self.aug_erase.is_enabled(), "p": self.aug_erase.values()[0]},
            },
        }

    def set_config(self, cfg: dict):
        crop = cfg.get("crop", {})
        self.chk_crop.setChecked(crop.get("enabled", False))
        if crop.get("type") == "center":
            self.rb_center.setChecked(True)
        else:
            self.rb_manual.setChecked(True)
        self.sp_cx.setValue(int(crop.get("x", 0)))
        self.sp_cy.setValue(int(crop.get("y", 0)))
        self.sp_cw.setValue(int(crop.get("w", 224)))
        self.sp_ch.setValue(int(crop.get("h", 224)))
        self.sp_center_size.setValue(int(crop.get("center_size", 224)))

        rz = cfg.get("resize", {})
        self.sp_rw.setValue(int(rz.get("width", 224)))
        self.sp_rh.setValue(int(rz.get("height", 224)))

        norm = cfg.get("normalize", {})
        self.chk_imagenet_norm.setChecked(norm.get("use_imagenet", True))
        self.edit_mean.setText(", ".join(str(v) for v in norm.get("mean", [0.485, 0.456, 0.406])))
        self.edit_std.setText(", ".join(str(v) for v in norm.get("std",  [0.229, 0.224, 0.225])))

        aug = cfg.get("augmentation", {})
        _set_aug(self.aug_hflip, aug.get("h_flip", {}), ["p"])
        _set_aug(self.aug_vflip, aug.get("v_flip", {}), ["p"])
        _set_aug(self.aug_rot,   aug.get("rotation", {}), ["degrees"])
        cj = aug.get("color_jitter", {})
        self.aug_cj.chk.setChecked(cj.get("enabled", False))
        for sp, key in zip(self.aug_cj.spins, ["brightness", "contrast", "saturation", "hue"]):
            sp.setValue(float(cj.get(key, sp.value())))
        rrc = aug.get("random_resized_crop", {})
        self.aug_rrc.chk.setChecked(rrc.get("enabled", False))
        for sp, key in zip(self.aug_rrc.spins, ["scale_min", "scale_max"]):
            sp.setValue(float(rrc.get(key, sp.value())))
        _set_aug(self.aug_blur,  aug.get("gaussian_blur", {}), ["kernel_size"])
        _set_aug(self.aug_erase, aug.get("random_erasing", {}), ["p"])
        self._update_crop_mode()


# ---------------------------------------------------------------- helpers
def _make_line(text):
    from qtpy.QtWidgets import QLineEdit
    ed = QLineEdit(text)
    ed.setMaximumWidth(240)
    return ed


def _section_label(text):
    lbl = QLabel(text)
    lbl.setFont(QFont("Arial", 9, QFont.Bold))
    lbl.setStyleSheet("color:#444; margin-top:4px;")
    return lbl


def _parse_floats(text, default):
    try:
        vals = [float(v.strip()) for v in text.split(",")]
        return vals if len(vals) == 3 else default
    except Exception:
        return default


def _set_aug(row: AugmentRow, cfg: dict, keys: list):
    row.chk.setChecked(cfg.get("enabled", False))
    for sp, key in zip(row.spins, keys):
        if key in cfg:
            sp.setValue(float(cfg[key]))
