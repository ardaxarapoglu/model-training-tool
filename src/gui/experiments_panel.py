"""Experiments panel: manage experiment list, forms, and split assignments."""
import os
import uuid

from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QFormLayout, QLineEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QGroupBox, QLabel, QSplitter, QHeaderView,
    QFileDialog, QMessageBox, QAbstractItemView, QDoubleSpinBox,
    QFrame, QToolBar, QSizePolicy,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFont


SPLITS = ["train", "validation", "test"]
# validation = watched during training (early stopping / LR scheduling)
# test       = held out, evaluated exactly once at the end
SPLIT_COLORS = {
    "train":      QColor("#c8e6c9"),   # green
    "validation": QColor("#fff9c4"),   # yellow  – active during training
    "test":       QColor("#b3e5fc"),   # blue    – final hold-out
}
N_FRAMES = 7


class ExperimentsPanel(QWidget):
    experiments_changed = Signal()

    def __init__(self):
        super().__init__()
        self._experiments = []
        self._current_idx = -1
        self._loading = False
        self._setup_ui()

    # ------------------------------------------------------------------ UI setup
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left: experiment list panel ----
        left = QWidget()
        left.setMaximumWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        hdr = QLabel("Experiments")
        hdr.setFont(QFont("Arial", 11, QFont.Bold))
        lv.addWidget(hdr)

        self.exp_list = QListWidget()
        self.exp_list.setAlternatingRowColors(True)
        self.exp_list.currentRowChanged.connect(self._on_row_changed)
        lv.addWidget(self.exp_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._add_experiment)
        self.btn_remove = QPushButton("– Remove")
        self.btn_remove.clicked.connect(self._remove_experiment)
        self.btn_remove.setEnabled(False)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        lv.addLayout(btn_row)

        self.btn_import = QPushButton("Import from forms/ + frames/")
        self.btn_import.setToolTip(
            "Scan the forms/ and frames/ directories next to this project "
            "and auto-import all experiments."
        )
        self.btn_import.clicked.connect(self._import_all)
        lv.addWidget(self.btn_import)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        lv.addWidget(sep)

        self.lbl_summary = QLabel("Train: 0  Val: 0  Test: 0")
        self.lbl_summary.setStyleSheet("color:#555;font-size:11px;")
        lv.addWidget(self.lbl_summary)

        splitter.addWidget(left)

        # ---- Right: detail form ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(6)

        self.grp_form = QGroupBox("Experiment Details")
        self.grp_form.setEnabled(False)
        fv = QVBoxLayout(self.grp_form)

        basic = QFormLayout()
        basic.setLabelAlignment(Qt.AlignRight)
        basic.setHorizontalSpacing(10)

        self.edit_name = QLineEdit()
        self.edit_name.textChanged.connect(self._save_current)
        basic.addRow("Name:", self.edit_name)

        self.edit_exp_no = QLineEdit()
        self.edit_exp_no.setReadOnly(True)
        self.edit_exp_no.setStyleSheet("background:#f5f5f5;")
        basic.addRow("Exp. No:", self.edit_exp_no)

        self.edit_date = QLineEdit()
        self.edit_date.setReadOnly(True)
        self.edit_date.setStyleSheet("background:#f5f5f5;")
        basic.addRow("Date:", self.edit_date)

        self.edit_operator = QLineEdit()
        self.edit_operator.setReadOnly(True)
        self.edit_operator.setStyleSheet("background:#f5f5f5;")
        basic.addRow("Operator:", self.edit_operator)

        self.cmb_split = QComboBox()
        _SPLIT_LABELS = {"train": "Train", "validation": "Validation (monitored)", "test": "Test (held out)"}
        for s in SPLITS:
            self.cmb_split.addItem(_SPLIT_LABELS[s], s)
        self.cmb_split.setToolTip(
            "Train      – images used to update model weights every epoch.\n"
            "Validation – monitored after each epoch for early stopping and LR scheduling.\n"
            "Test       – NEVER seen during training; evaluated once for the final report."
        )
        self.cmb_split.currentIndexChanged.connect(self._save_current)
        basic.addRow("Assigned to:", self.cmb_split)

        self.edit_notes = QLineEdit()
        self.edit_notes.setPlaceholderText("Optional notes")
        self.edit_notes.textChanged.connect(self._save_current)
        basic.addRow("Notes:", self.edit_notes)

        fv.addLayout(basic)

        # Form path row
        form_row = QHBoxLayout()
        self.edit_form_path = QLineEdit()
        self.edit_form_path.setPlaceholderText("Path to .xlsx form file")
        self.edit_form_path.textChanged.connect(self._save_current)
        btn_browse_form = QPushButton("Browse…")
        btn_browse_form.clicked.connect(self._browse_form)
        btn_reimport = QPushButton("Re-import labels")
        btn_reimport.setToolTip("Re-read PB values from the xlsx form")
        btn_reimport.clicked.connect(self._reimport_form)
        form_row.addWidget(self.edit_form_path)
        form_row.addWidget(btn_browse_form)
        form_row.addWidget(btn_reimport)
        fv.addLayout(form_row)

        # Time frames table
        tf_lbl = QLabel("Time Frames")
        tf_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        fv.addWidget(tf_lbl)

        self.tf_table = QTableWidget(N_FRAMES, 5)
        self.tf_table.setHorizontalHeaderLabels(
            ["Frame", "Time Interval", "Folder", "Pb Tenörü (%)", "Notes"]
        )
        hh = self.tf_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tf_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tf_table.setMinimumHeight(220)

        for i in range(N_FRAMES):
            name_item = QTableWidgetItem(f"Y{i+1}")
            name_item.setFlags(Qt.ItemIsEnabled)
            self.tf_table.setItem(i, 0, name_item)

            self.tf_table.setItem(i, 1, QTableWidgetItem(""))

            folder_w = self._make_folder_widget(i)
            self.tf_table.setCellWidget(i, 2, folder_w)

            pb_spin = QDoubleSpinBox()
            pb_spin.setRange(0.0, 10000.0)
            pb_spin.setDecimals(4)
            pb_spin.setSingleStep(0.1)
            pb_spin.valueChanged.connect(self._save_current)
            self.tf_table.setCellWidget(i, 3, pb_spin)

            self.tf_table.setItem(i, 4, QTableWidgetItem(""))

        self.tf_table.itemChanged.connect(self._save_current)
        fv.addWidget(self.tf_table)

        auto_row = QHBoxLayout()
        btn_auto = QPushButton("Auto-detect folders from root")
        btn_auto.setToolTip(
            "Set the experiment root folder and this will scan for Y1–Y7 sub-folders."
        )
        btn_auto.clicked.connect(self._auto_detect)
        auto_row.addWidget(btn_auto)
        auto_row.addStretch()
        fv.addLayout(auto_row)

        rv.addWidget(self.grp_form)
        rv.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([260, 900])

    # ---------------------------------------------------------------- folder widget
    def _make_folder_widget(self, row):
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(2, 1, 2, 1)
        ed = QLineEdit()
        ed.setPlaceholderText("path…")
        ed.textChanged.connect(self._save_current)
        btn = QPushButton("…")
        btn.setFixedWidth(26)
        btn.clicked.connect(lambda _, r=row: self._browse_tf(r))
        hl.addWidget(ed)
        hl.addWidget(btn)
        return w

    def _tf_edit(self, row):
        cw = self.tf_table.cellWidget(row, 2)
        return cw.layout().itemAt(0).widget() if cw else None

    def _tf_spin(self, row):
        return self.tf_table.cellWidget(row, 3)

    # ---------------------------------------------------------------- actions
    def _browse_tf(self, row):
        path = QFileDialog.getExistingDirectory(self, f"Select folder for Y{row+1}")
        if path:
            ed = self._tf_edit(row)
            if ed:
                ed.setText(path)

    def _browse_form(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select form xlsx", "", "Excel (*.xlsx)")
        if path:
            self.edit_form_path.setText(path)

    def _reimport_form(self):
        if self._current_idx < 0:
            return
        path = self.edit_form_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No Form", "Set a valid xlsx form path first.")
            return
        from ..utils.form_parser import parse_experiment_form
        try:
            info = parse_experiment_form(path)
        except Exception as e:
            QMessageBox.critical(self, "Parse Error", str(e))
            return
        self._apply_form_info(info)
        self._save_current()

    def _auto_detect(self):
        if self._current_idx < 0:
            return
        root, _ = QFileDialog.getExistingDirectory(self, "Select experiment root folder"), None
        if not root:
            root = QFileDialog.getExistingDirectory(self, "Select experiment root folder")
        if not root:
            return
        subdirs = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        for i, d in enumerate(subdirs[:N_FRAMES]):
            ed = self._tf_edit(i)
            if ed:
                ed.setText(os.path.join(root, d))

    def _import_all(self):
        base = QFileDialog.getExistingDirectory(
            self, "Select project root (containing forms/ and frames/ folders)"
        )
        if not base:
            return
        forms_dir = os.path.join(base, "forms")
        frames_dir = os.path.join(base, "frames")
        if not os.path.isdir(forms_dir):
            QMessageBox.warning(self, "Not Found", f"No forms/ directory in {base}")
            return
        from ..utils.form_parser import discover_experiments
        new_exps = discover_experiments(forms_dir, frames_dir)
        if not new_exps:
            QMessageBox.information(self, "No Experiments", "No xlsx forms found.")
            return

        # Merge: add only experiments whose id is not already present
        existing_ids = {e["id"] for e in self._experiments}
        added = 0
        for exp in new_exps:
            if exp["id"] not in existing_ids:
                self._experiments.append(exp)
                self._add_list_item(exp)
                added += 1

        self._update_summary()
        QMessageBox.information(
            self, "Import Done",
            f"Imported {added} new experiment(s). "
            f"Total: {len(self._experiments)}."
        )
        if self._experiments and self.exp_list.currentRow() < 0:
            self.exp_list.setCurrentRow(0)

    def _add_experiment(self):
        exp = self._blank_experiment()
        self._experiments.append(exp)
        self._add_list_item(exp)
        self.exp_list.setCurrentRow(len(self._experiments) - 1)
        self._update_summary()

    def _blank_experiment(self):
        n = len(self._experiments) + 1
        return {
            "id": str(uuid.uuid4()),
            "name": f"Experiment {n}",
            "form_path": "",
            "experiment_no": "",
            "date": "",
            "operator": "",
            "split": "train",
            "notes": "",
            "time_frames": [
                {
                    "name": f"Y{i+1}",
                    "time_interval": "",
                    "pb_concentration": 0.0,
                    "pb_distribution": 0.0,
                    "folder_path": "",
                    "notes": "",
                }
                for i in range(N_FRAMES)
            ],
        }

    def _add_list_item(self, exp):
        item = QListWidgetItem(exp["name"])
        item.setData(Qt.UserRole, exp["id"])
        item.setBackground(SPLIT_COLORS.get(exp.get("split", "train"), QColor("white")))
        self.exp_list.addItem(item)

    def _remove_experiment(self):
        row = self.exp_list.currentRow()
        if row < 0:
            return
        name = self._experiments[row]["name"]
        if QMessageBox.question(
            self, "Remove", f"Remove '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self._experiments.pop(row)
        self.exp_list.takeItem(row)
        self._current_idx = -1
        if self._experiments:
            self.exp_list.setCurrentRow(min(row, len(self._experiments) - 1))
        else:
            self.grp_form.setEnabled(False)
            self.btn_remove.setEnabled(False)
        self._update_summary()

    # ---------------------------------------------------------------- load/save form
    def _on_row_changed(self, row):
        self._current_idx = row
        if row < 0 or row >= len(self._experiments):
            self.grp_form.setEnabled(False)
            self.btn_remove.setEnabled(False)
            return
        self.grp_form.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self._loading = True
        self._load_to_form(self._experiments[row])
        self._loading = False

    def _load_to_form(self, exp):
        self.edit_name.setText(exp.get("name", ""))
        self.edit_exp_no.setText(exp.get("experiment_no", ""))
        self.edit_date.setText(exp.get("date", ""))
        self.edit_operator.setText(exp.get("operator", ""))
        self.edit_form_path.setText(exp.get("form_path", ""))
        self.edit_notes.setText(exp.get("notes", ""))

        split_idx = self.cmb_split.findData(exp.get("split", "train"))
        self.cmb_split.setCurrentIndex(max(0, split_idx))

        for i, tf in enumerate(exp.get("time_frames", [])[:N_FRAMES]):
            item_interval = self.tf_table.item(i, 1)
            if item_interval:
                item_interval.setText(tf.get("time_interval", ""))

            ed = self._tf_edit(i)
            if ed:
                ed.blockSignals(True)
                ed.setText(tf.get("folder_path", ""))
                ed.blockSignals(False)

            sp = self._tf_spin(i)
            if sp:
                sp.blockSignals(True)
                sp.setValue(float(tf.get("pb_concentration", 0.0)))
                sp.blockSignals(False)

            item_notes = self.tf_table.item(i, 4)
            if item_notes:
                item_notes.setText(tf.get("notes", ""))

    def _apply_form_info(self, info: dict):
        """Apply parsed form data to the current UI state."""
        self.edit_exp_no.setText(info.get("experiment_no", ""))
        self.edit_date.setText(info.get("date", ""))
        self.edit_operator.setText(info.get("operator", ""))
        for i, tf in enumerate(info.get("time_frames", [])[:N_FRAMES]):
            item = self.tf_table.item(i, 1)
            if item:
                item.setText(tf.get("time_interval", ""))
            sp = self._tf_spin(i)
            if sp:
                sp.setValue(float(tf.get("pb_concentration", 0.0)))

    def _save_current(self, *_):
        if self._loading:
            return
        if self._current_idx < 0 or self._current_idx >= len(self._experiments):
            return
        exp = self._experiments[self._current_idx]
        exp["name"] = self.edit_name.text()
        exp["form_path"] = self.edit_form_path.text()
        exp["notes"] = self.edit_notes.text()
        exp["split"] = self.cmb_split.currentData()

        for i in range(N_FRAMES):
            tf = exp["time_frames"][i]
            item_iv = self.tf_table.item(i, 1)
            if item_iv:
                tf["time_interval"] = item_iv.text()
            ed = self._tf_edit(i)
            if ed:
                tf["folder_path"] = ed.text()
            sp = self._tf_spin(i)
            if sp:
                tf["pb_concentration"] = sp.value()
            item_n = self.tf_table.item(i, 4)
            if item_n:
                tf["notes"] = item_n.text()

        # Refresh list item
        item = self.exp_list.item(self._current_idx)
        if item:
            item.setText(exp["name"])
            item.setBackground(SPLIT_COLORS.get(exp["split"], QColor("white")))

        self._update_summary()
        self.experiments_changed.emit()

    def _update_summary(self):
        from collections import Counter
        c = Counter(e.get("split", "train") for e in self._experiments)
        self.lbl_summary.setText(
            f"Train: {c['train']}  Val: {c['validation']}  Test: {c['test']}"
        )

    # ---------------------------------------------------------------- public API
    def get_config(self) -> list:
        self._save_current()
        return list(self._experiments)

    def set_config(self, experiments: list):
        self._experiments = []
        self.exp_list.clear()
        self._current_idx = -1
        self.grp_form.setEnabled(False)

        for exp in experiments:
            if "id" not in exp:
                exp["id"] = str(uuid.uuid4())
            tfs = exp.setdefault("time_frames", [])
            while len(tfs) < N_FRAMES:
                tfs.append({
                    "name": f"Y{len(tfs)+1}",
                    "time_interval": "", "pb_concentration": 0.0,
                    "pb_distribution": 0.0, "folder_path": "", "notes": "",
                })
            exp["time_frames"] = tfs[:N_FRAMES]
            self._experiments.append(exp)
            self._add_list_item(exp)

        self._update_summary()
        if self._experiments:
            self.exp_list.setCurrentRow(0)
