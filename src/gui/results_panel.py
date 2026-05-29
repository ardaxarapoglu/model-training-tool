"""Results panel: table of all runs + loss curves + confusion matrix + performance evaluation."""
import os
import csv
import math

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QSplitter, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QTabWidget, QProgressBar, QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


RESULT_COLS = [
    "Run ID", "Mode", "Architecture", "Batch", "LR", "Optimizer", "WD", "Loss Fn",
    "Best Ep.", "Train Loss",
    # Validation = monitored during training (early stopping / LR scheduling)
    "Val Loss", "Val RMSE", "Val MAE", "Val R²", "Val Acc", "Val F1",
    # Test = held-out, evaluated exactly once after training completes
    "Test RMSE", "Test MAE", "Test R²", "Test Acc", "Test F1",
    "Time (s)", "Checkpoint",
]

# ── Confusion-matrix column header tooltips ───────────────────────────────
_CM_HEADERS = [
    ("Class",
     "The class name (e.g. Bad / Acceptable / Good)."),
    ("Support",
     "Number of actual images of this class in the evaluation set.\n"
     "Low support means this class's metrics are less reliable —\n"
     "a single mis-classification can swing the score dramatically."),
    ("Precision",
     "Of all images the model predicted as this class,\n"
     "the fraction that were actually correct.\n"
     "= TP / (TP + FP)\n"
     "Low precision → many false alarms for this class."),
    ("Recall",
     "Of all actual images of this class,\n"
     "the fraction the model correctly identified.\n"
     "= TP / (TP + FN)\n"
     "Low recall → the model misses many real cases of this class."),
    ("F1",
     "Harmonic mean of Precision and Recall.\n"
     "= 2 × Precision × Recall / (Precision + Recall)\n"
     "Range 0–1; balances both metrics into one score.\n"
     "Colour: green ≥ 0.70 | yellow ≥ 0.40 | red < 0.40"),
]

_PERF_HDRS = [
    ("Metric",           "Name of the evaluation metric"),
    ("Value",            "Computed value for this run"),
    ("Rating",           "Qualitative interpretation (see Metric Glossary tab)"),
    ("What it measures", "One-line description of what this metric captures"),
]

_PERF_BRIEF = {
    "Accuracy":          "Correct predictions / total images",
    "Balanced Accuracy": "Mean recall per class — robust to class imbalance",
    "Cohen's Kappa":     "Agreement corrected for chance  (0=random, 1=perfect)",
    "MCC":               "Correlation coeff. for multi-class  (−1 to +1)",
    "Macro F1":          "Mean F1 per class — equal weight to all classes",
    "Weighted F1":       "Mean F1 per class — weighted by class sample count",
}


def _compute_cls_perf_metrics(preds: list, labels: list, n_classes: int) -> dict:
    """Compute comprehensive classification performance metrics from predictions and labels."""
    p = np.array(preds,  dtype=int)
    y = np.array(labels, dtype=int)

    # Confusion matrix (row = true, col = predicted)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred_i, true_i in zip(p, y):
        if 0 <= true_i < n_classes and 0 <= pred_i < n_classes:
            cm[true_i, pred_i] += 1

    total   = len(p)
    correct = int(np.diag(cm).sum())
    accuracy = correct / max(total, 1)

    # Balanced accuracy = mean per-class recall
    per_class_recall = []
    for i in range(n_classes):
        s = cm[i, :].sum()
        per_class_recall.append(float(cm[i, i]) / s if s > 0 else 0.0)
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    # Cohen's Kappa (multi-class)
    po = accuracy
    pe = sum(
        (int(cm[i, :].sum()) * int(cm[:, i].sum())) for i in range(n_classes)
    ) / (total ** 2) if total > 0 else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-9 else 0.0

    # Matthews Correlation Coefficient (Gorodkin multi-class formula)
    t_k = np.sum(cm, axis=1).astype(float)   # actual counts per class
    p_k = np.sum(cm, axis=0).astype(float)   # predicted counts per class
    c   = float(np.diag(cm).sum())
    s   = float(cm.sum())
    mcc_num = c * s - float(np.dot(t_k, p_k))
    mcc_den = np.sqrt(
        (s ** 2 - float(np.dot(p_k, p_k))) *
        (s ** 2 - float(np.dot(t_k, t_k)))
    )
    mcc = mcc_num / mcc_den if mcc_den > 1e-9 else 0.0

    # Macro F1 and Weighted F1
    macro_f1_sum    = 0.0
    weighted_f1_sum = 0.0
    for i in range(n_classes):
        tp      = cm[i, i]
        fp      = cm[:, i].sum() - tp
        fn      = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec    = tp / (tp + fp + 1e-9)
        rec     = tp / (tp + fn + 1e-9)
        f1      = 2 * prec * rec / (prec + rec + 1e-9)
        macro_f1_sum    += f1
        weighted_f1_sum += f1 * support

    macro_f1    = macro_f1_sum    / n_classes if n_classes > 0 else 0.0
    weighted_f1 = weighted_f1_sum / max(total, 1)

    return {
        "accuracy":          accuracy,
        "balanced_accuracy": balanced_acc,
        "kappa":             kappa,
        "mcc":               float(mcc),
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "cm":                cm,
    }


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._results    = []
        self._output_dir = "./results"
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        self.btn_clear  = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_load   = QPushButton("Load Previous Runs…")
        self.btn_load.setToolTip("Load saved result.json files from a results directory")
        self.btn_load.clicked.connect(self._load_previous)
        self.lbl_best   = QLabel("")
        self.lbl_best.setStyleSheet("font-weight:bold;color:#1b5e20;")
        toolbar.addWidget(self.btn_clear)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.lbl_best)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter)

        # ── Runs table ───────────────────────────────────────────────────
        self.table = QTableWidget(0, len(RESULT_COLS))
        self.table.setHorizontalHeaderLabels(RESULT_COLS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        # ── Chart / detail tabs ──────────────────────────────────────────
        self.chart_tabs = QTabWidget()
        splitter.addWidget(self.chart_tabs)

        # ── Tab 0: Overview (Confusion Matrix + Performance side-by-side) ─
        combo_w     = QWidget()
        combo_outer = QVBoxLayout(combo_w)
        combo_outer.setContentsMargins(2, 2, 2, 2)
        combo_split = QSplitter(Qt.Horizontal)
        combo_outer.addWidget(combo_split)

        # Left — confusion matrix plot
        combo_cm_w = QWidget()
        combo_cm_v = QVBoxLayout(combo_cm_w)
        combo_cm_v.setContentsMargins(0, 0, 0, 0)
        self.fig_cm_combo    = Figure(figsize=(4, 4), tight_layout=True)
        self.ax_cm_combo     = self.fig_cm_combo.add_subplot(111)
        self.canvas_cm_combo = FigureCanvas(self.fig_cm_combo)
        combo_cm_v.addWidget(self.canvas_cm_combo)
        combo_split.addWidget(combo_cm_w)

        # Right — performance summary
        combo_perf_w = QWidget()
        combo_perf_v = QVBoxLayout(combo_perf_w)
        combo_perf_v.setContentsMargins(6, 4, 4, 4)
        combo_perf_v.setSpacing(6)

        self.grp_thesis_combo = QGroupBox("Accuracy  (≥ 70% target)")
        thesis_combo_h = QHBoxLayout(self.grp_thesis_combo)
        thesis_combo_h.setSpacing(12)
        self.lbl_thesis_acc_combo = QLabel("—")
        self.lbl_thesis_acc_combo.setFont(QFont("Arial", 36, QFont.Bold))
        self.lbl_thesis_acc_combo.setAlignment(Qt.AlignCenter)
        self.lbl_thesis_acc_combo.setStyleSheet("color:#555;")
        self.lbl_thesis_acc_combo.setMinimumWidth(110)
        thesis_combo_h.addWidget(self.lbl_thesis_acc_combo)
        self.lbl_thesis_result_combo = QLabel("Select a run to evaluate.")
        self.lbl_thesis_result_combo.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#555;"
        )
        self.lbl_thesis_result_combo.setWordWrap(True)
        thesis_combo_h.addWidget(self.lbl_thesis_result_combo, 1)
        combo_perf_v.addWidget(self.grp_thesis_combo)

        self.grp_metrics_combo = QGroupBox("Classification Metrics")
        summary_combo_v = QVBoxLayout(self.grp_metrics_combo)
        summary_combo_v.setContentsMargins(4, 4, 4, 4)
        self.tbl_perf_combo = QTableWidget(0, 4)
        for col, (lbl, tip) in enumerate(_PERF_HDRS):
            hi = QTableWidgetItem(lbl)
            hi.setToolTip(tip)
            self.tbl_perf_combo.setHorizontalHeaderItem(col, hi)
        self.tbl_perf_combo.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_perf_combo.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_perf_combo.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_perf_combo.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl_perf_combo.verticalHeader().setVisible(False)
        self.tbl_perf_combo.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_perf_combo.setAlternatingRowColors(True)
        summary_combo_v.addWidget(self.tbl_perf_combo)
        combo_perf_v.addWidget(self.grp_metrics_combo)

        _cls_hdr_combo = QLabel("Per-class Metrics")
        _cls_hdr_combo.setFont(QFont("Arial", 9, QFont.Bold))
        combo_perf_v.addWidget(_cls_hdr_combo)

        self.tbl_cls_metrics_combo = QTableWidget(0, 5)
        for col, (label, tip) in enumerate(_CM_HEADERS):
            hitem = QTableWidgetItem(label)
            hitem.setToolTip(tip)
            self.tbl_cls_metrics_combo.setHorizontalHeaderItem(col, hitem)
        self.tbl_cls_metrics_combo.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.tbl_cls_metrics_combo.verticalHeader().setVisible(False)
        self.tbl_cls_metrics_combo.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_cls_metrics_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        combo_perf_v.addWidget(self.tbl_cls_metrics_combo, 1)

        self.lbl_overall_metrics_combo = QLabel("")
        self.lbl_overall_metrics_combo.setStyleSheet(
            "font-size:10px; color:#333; margin-top:2px;"
        )
        self.lbl_overall_metrics_combo.setWordWrap(True)
        combo_perf_v.addWidget(self.lbl_overall_metrics_combo)

        combo_split.addWidget(combo_perf_w)
        combo_split.setSizes([430, 370])
        self.chart_tabs.addTab(combo_w, "Overview")

        # ── Tab 1: Loss Curves ────────────────────────────────────────────
        loss_w      = QWidget()
        loss_layout = QVBoxLayout(loss_w)
        loss_layout.setContentsMargins(2, 2, 2, 2)
        self.fig_loss  = Figure(figsize=(6, 3), tight_layout=True)
        self.ax_loss   = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        loss_layout.addWidget(self.canvas_loss)
        self.chart_tabs.addTab(loss_w, "Loss Curves")

        # ── Tab 2: Confusion Matrix + per-class metrics ───────────────────
        cm_w     = QWidget()
        cm_outer = QVBoxLayout(cm_w)
        cm_outer.setContentsMargins(2, 2, 2, 2)

        cm_split = QSplitter(Qt.Horizontal)
        cm_outer.addWidget(cm_split)

        # Left — confusion matrix plot
        cm_plot_w = QWidget()
        cm_plot_v = QVBoxLayout(cm_plot_w)
        cm_plot_v.setContentsMargins(0, 0, 0, 0)
        self.fig_cm    = Figure(figsize=(4, 4), tight_layout=True)
        self.ax_cm     = self.fig_cm.add_subplot(111)
        self.canvas_cm = FigureCanvas(self.fig_cm)
        cm_plot_v.addWidget(self.canvas_cm)
        cm_split.addWidget(cm_plot_w)

        # Right — per-class metrics (expands to fill full height)
        cls_panel = QWidget()
        cls_v     = QVBoxLayout(cls_panel)
        cls_v.setContentsMargins(6, 4, 4, 4)
        cls_v.setSpacing(4)

        cls_hdr = QLabel("Per-class Metrics")
        cls_hdr.setFont(QFont("Arial", 9, QFont.Bold))
        cls_v.addWidget(cls_hdr)

        self.tbl_cls_metrics = QTableWidget(0, 5)
        for col, (label, tip) in enumerate(_CM_HEADERS):
            hitem = QTableWidgetItem(label)
            hitem.setToolTip(tip)
            self.tbl_cls_metrics.setHorizontalHeaderItem(col, hitem)
        self.tbl_cls_metrics.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.tbl_cls_metrics.verticalHeader().setVisible(False)
        self.tbl_cls_metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_cls_metrics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cls_v.addWidget(self.tbl_cls_metrics, 1)   # stretch=1 → fills remaining height

        self.lbl_overall_metrics = QLabel("")
        self.lbl_overall_metrics.setStyleSheet(
            "font-size:11px; color:#333; margin-top:2px;"
        )
        self.lbl_overall_metrics.setWordWrap(True)
        cls_v.addWidget(self.lbl_overall_metrics)

        # Metric definition legend (plain text, always visible)
        defs_lbl = QLabel(
            "<b>Definitions</b> (hover column headers for details)<br>"
            "<b>Support</b> — images of this class in the eval set<br>"
            "<b>Precision</b> — correct / all predicted as this class  (TP/(TP+FP))<br>"
            "<b>Recall</b> — correct / all actual of this class  (TP/(TP+FN))<br>"
            "<b>F1</b> — harmonic mean of Precision &amp; Recall<br>"
            "<b>Accuracy</b> — overall correct / total  (shown in summary above)<br>"
            "<span style='color:#2e7d32'>■</span> F1 ≥ 0.70 good &nbsp;"
            "<span style='color:#f57f17'>■</span> ≥ 0.40 moderate &nbsp;"
            "<span style='color:#b71c1c'>■</span> &lt; 0.40 poor"
        )
        defs_lbl.setWordWrap(True)
        defs_lbl.setTextFormat(Qt.RichText)
        defs_lbl.setStyleSheet(
            "font-size:10px; color:#555; background:#f9f9f9;"
            "border:1px solid #ddd; border-radius:3px; padding:4px; margin-top:4px;"
        )
        cls_v.addWidget(defs_lbl)

        cm_split.addWidget(cls_panel)
        cm_split.setSizes([390, 320])
        self.chart_tabs.addTab(cm_w, "Confusion Matrix")

        # ── Tab 3: Performance Evaluation ─────────────────────────────────
        perf_w   = QWidget()
        perf_v   = QVBoxLayout(perf_w)
        perf_v.setContentsMargins(8, 8, 8, 8)
        perf_v.setSpacing(8)

        # ── Accuracy section (compact) ────────────────────────────────────
        self.grp_thesis = QGroupBox("Accuracy  (≥ 70% target)")
        thesis_v = QHBoxLayout(self.grp_thesis)   # horizontal: big number | verdict
        thesis_v.setSpacing(16)

        self.lbl_thesis_acc = QLabel("—")
        self.lbl_thesis_acc.setFont(QFont("Arial", 46, QFont.Bold))
        self.lbl_thesis_acc.setAlignment(Qt.AlignCenter)
        self.lbl_thesis_acc.setStyleSheet("color:#555;")
        self.lbl_thesis_acc.setMinimumWidth(140)
        thesis_v.addWidget(self.lbl_thesis_acc)

        self.lbl_thesis_result = QLabel("Select a run to evaluate.")
        self.lbl_thesis_result.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#555;"
        )
        self.lbl_thesis_result.setWordWrap(True)
        thesis_v.addWidget(self.lbl_thesis_result, 1)

        perf_v.addWidget(self.grp_thesis)

        # ── Summary metrics table (expands to fill remaining space) ───────
        self.grp_metrics_summary = QGroupBox(
            "Classification Metrics  (Test Set — Held Out)"
        )
        summary_v = QVBoxLayout(self.grp_metrics_summary)
        summary_v.setContentsMargins(4, 4, 4, 4)

        self.tbl_perf = QTableWidget(0, 4)
        for col, (lbl, tip) in enumerate(_PERF_HDRS):
            hi = QTableWidgetItem(lbl)
            hi.setToolTip(tip)
            self.tbl_perf.setHorizontalHeaderItem(col, hi)
        self.tbl_perf.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_perf.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_perf.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_perf.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl_perf.verticalHeader().setVisible(False)
        self.tbl_perf.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_perf.setAlternatingRowColors(True)
        self.tbl_perf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        summary_v.addWidget(self.tbl_perf)
        perf_v.addWidget(self.grp_metrics_summary, 1)   # stretch=1 → fills remainder

        self.chart_tabs.addTab(perf_w, "Performance Evaluation")

        # ── Tab 4: Metric Glossary ─────────────────────────────────────────
        glossary_w      = QWidget()
        glossary_outer  = QVBoxLayout(glossary_w)
        glossary_outer.setContentsMargins(2, 2, 2, 2)

        glossary_scroll = QScrollArea()
        glossary_scroll.setWidgetResizable(True)
        glossary_scroll.setFrameShape(QFrame.NoFrame)
        glossary_outer.addWidget(glossary_scroll)

        glossary_container = QWidget()
        glossary_inner     = QVBoxLayout(glossary_container)
        glossary_inner.setContentsMargins(12, 10, 12, 10)
        glossary_scroll.setWidget(glossary_container)

        glossary_lbl = QLabel(
            "<b style='font-size:13px'>Accuracy</b><br>"
            "Overall percentage of images classified into the correct class.  "
            "= (correctly classified) / (total images).  "
            "This is the primary target: the model must reach ≥ 70%.<br><br>"

            "<b style='font-size:13px'>Balanced Accuracy</b><br>"
            "Average recall across all classes, each class weighted equally — "
            "unlike plain accuracy, it is not fooled by class imbalance.  "
            "Example: if the model ignores a rare class, plain accuracy stays high "
            "while balanced accuracy exposes the failure.<br><br>"

            "<b style='font-size:13px'>Cohen's Kappa (κ)</b><br>"
            "How much better than chance the model is, accounting for the class "
            "frequency distribution.  κ = 0 means no better than a random guesser; "
            "κ = 1 means perfect.  "
            "Bands: ≥ 0.81 Almost perfect | 0.61–0.80 Substantial | "
            "0.41–0.60 Moderate | 0.21–0.40 Fair | ≤ 0.20 Slight or none.<br><br>"

            "<b style='font-size:13px'>Matthews Correlation Coefficient (MCC)</b><br>"
            "A correlation coefficient between the true and predicted labels.  "
            "Handles multi-class problems and class imbalance better than accuracy or F1.  "
            "Range − 1 (all wrong) to + 1 (perfect); 0 means random.  "
            "Often considered the most informative single number for classification.<br><br>"

            "<b style='font-size:13px'>Macro F1</b><br>"
            "F1 computed separately per class then averaged with equal weight.  "
            "Forces good performance on every class regardless of size.  "
            "Use this when all class errors are equally costly.<br><br>"

            "<b style='font-size:13px'>Weighted F1</b><br>"
            "Same as Macro F1 but each class's F1 is weighted by its sample count.  "
            "Reflects real-world performance when class sizes differ.  "
            "Use this for an overall realistic picture.<br><br>"

            "<b style='font-size:13px'>Precision (per class)</b><br>"
            "Of all images the model predicted as this class, "
            "the fraction that were actually correct.  "
            "= TP / (TP + FP).  "
            "Low precision → many false alarms (model over-predicts this class).<br><br>"

            "<b style='font-size:13px'>Recall (per class)</b><br>"
            "Of all actual images of this class, "
            "the fraction the model correctly found.  "
            "= TP / (TP + FN).  "
            "Low recall → the model frequently misses this class.<br><br>"

            "<b style='font-size:13px'>F1 (per class)</b><br>"
            "Harmonic mean of Precision and Recall: 2 × P × R / (P + R).  "
            "Low if either metric is low — you need both to be good at the same time.<br><br>"

            "<b style='font-size:13px'>Support</b><br>"
            "Number of actual images of this class in the evaluation set.  "
            "Classes with very low support have unreliable per-class metrics — "
            "a single additional correct or incorrect prediction can shift them dramatically.<br><br>"

            "<b style='font-size:13px'>RMSE</b>  (regression only)<br>"
            "Root Mean Squared Error — average prediction error in Pb% units.  "
            "Penalises large errors more than small ones.  Lower is better.<br><br>"

            "<b style='font-size:13px'>MAE</b>  (regression only)<br>"
            "Mean Absolute Error — average absolute prediction error in Pb% units.  "
            "More robust to large outlier errors than RMSE.  Lower is better.<br><br>"

            "<b style='font-size:13px'>R²</b>  (regression only)<br>"
            "Coefficient of determination.  "
            "1.0 = perfect predictions; 0.0 = no better than predicting the mean; "
            "negative = worse than the mean baseline.  Higher is better."
        )
        glossary_lbl.setWordWrap(True)
        glossary_lbl.setTextFormat(Qt.RichText)
        glossary_lbl.setStyleSheet("font-size:11px; color:#333; line-height:160%;")
        glossary_inner.addWidget(glossary_lbl)
        glossary_inner.addStretch()

        self.chart_tabs.addTab(glossary_w, "Metric Glossary")

        splitter.setSizes([300, 500])
        self._draw_placeholder()

    # ------------------------------------------------------------------ public
    def set_output_dir(self, path: str):
        self._output_dir = path

    def add_results(self, results: list):
        for r in results:
            self._results.append(r)
            self._append_row(r)
            self._auto_save(r)
        self._update_best_label()

    def add_result(self, result: dict):
        self._results.append(result)
        self._append_row(result)
        self._auto_save(result)
        self._update_best_label()

    # ------------------------------------------------------------------ persistence
    def _auto_save(self, r: dict):
        from ..utils.results_saver import save_result
        out = self._output_dir or "./results"
        save_result(r, out)

    def _load_previous(self):
        from ..utils.results_saver import load_results
        path = QFileDialog.getExistingDirectory(
            self, "Select Results Directory", self._output_dir
        )
        if not path:
            return
        loaded = load_results(path)
        if not loaded:
            QMessageBox.information(
                self, "No Results", "No result.json files found in that directory."
            )
            return
        existing_ids = {r.get("run_id") for r in self._results}
        new_count = 0
        for r in loaded:
            if r.get("run_id") not in existing_ids:
                self._results.append(r)
                self._append_row(r)
                new_count += 1
        self._update_best_label()
        QMessageBox.information(
            self, "Loaded", f"Loaded {new_count} new run(s) from:\n{path}"
        )

    # ------------------------------------------------------------------ table
    def _append_row(self, r: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        params = r.get("params", {})
        vm = r.get("final_val_metrics",  {})   # validation (monitored)
        tm = r.get("final_test_metrics", {})   # test (held-out)

        def _val(key, fmt="{:.4f}"):
            v = vm.get(key)
            return fmt.format(v) if v is not None else "—"

        def _tst(key, fmt="{:.4f}"):
            v = tm.get(key)
            return fmt.format(v) if v is not None else "—"

        train_hist = r.get("train_history", [])
        best_train = min(train_hist) if train_hist else None
        val_hist   = r.get("val_history", [])
        best_val   = min((v for v in val_hist if v is not None), default=None)

        values = [
            r.get("run_id", ""),
            r.get("mode", "regression"),
            str(params.get("architecture", "—")),
            str(params.get("batch_size",   "—")),
            str(params.get("learning_rate","—")),
            str(params.get("optimizer",    "—")),
            str(params.get("weight_decay", "—")),
            str(params.get("loss",         "—")),
            str(r.get("best_epoch", "—")),
            f"{best_train:.4f}" if best_train is not None else "—",
            # Validation
            f"{best_val:.4f}"        if best_val  is not None else "—",
            _val("rmse"),
            _val("mae"),
            _val("r2"),
            _val("accuracy", "{:.3f}"),
            _val("f1",       "{:.3f}"),
            # Test
            _tst("rmse"),
            _tst("mae"),
            _tst("r2"),
            _tst("accuracy", "{:.3f}"),
            _tst("f1",       "{:.3f}"),
            f"{r.get('elapsed_seconds', 0):.1f}",
            os.path.basename(r.get("checkpoint_path", "")),
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        if "error" in r:
            for col in range(len(RESULT_COLS)):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QColor("#ffcdd2"))

    def _update_best_label(self):
        best = self._find_best()
        if best:
            tm  = best.get("final_test_metrics") or best.get("final_val_metrics", {})
            src = "Test" if best.get("final_test_metrics") else "Val"
            if best.get("mode") == "classification":
                acc = tm.get("accuracy")
                metric_str = f"{src} Acc={acc:.3f}" if acc is not None else ""
            else:
                rmse = tm.get("rmse")
                metric_str = f"{src} RMSE={rmse:.4f}" if rmse is not None else ""
            self.lbl_best.setText(
                f"Best: {best.get('run_id', '')}  {metric_str}"
            )
        else:
            self.lbl_best.setText("")

        for row in range(self.table.rowCount()):
            is_best = (
                best and
                self.table.item(row, 0) and
                self.table.item(row, 0).text() == best.get("run_id", "")
            )
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and not item.background().color().isValid():
                    item.setBackground(
                        QColor("#e8f5e9") if is_best else QColor("transparent")
                    )

    def _find_best(self):
        valid = [r for r in self._results if "error" not in r]
        if not valid:
            return None
        def _metrics(r):
            return r.get("final_test_metrics") or r.get("final_val_metrics", {})
        if valid[0].get("mode") == "classification":
            return max(valid, key=lambda r: _metrics(r).get("accuracy", 0.0))
        return min(valid, key=lambda r: _metrics(r).get("rmse", math.inf))

    def _clear_all(self):
        if QMessageBox.question(
            self, "Clear", "Clear all results?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self._results.clear()
            self.table.setRowCount(0)
            self.lbl_best.setText("")
            self._draw_placeholder()

    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "No Results", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RESULT_COLS)
            for row in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(row, col).text()
                    if self.table.item(row, col) else ""
                    for col in range(self.table.columnCount())
                ])
        QMessageBox.information(self, "Exported", f"Results saved to {path}")

    # ------------------------------------------------------------------ selection
    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row_idx = selected[0].row()
        if row_idx < len(self._results):
            self._plot_result(self._results[row_idx])

    def _plot_result(self, r: dict):
        self._draw_loss_curves(r)
        self._draw_confusion_matrix(r)
        self._draw_performance_eval(r)

    # ------------------------------------------------------------------ loss curves
    def _draw_loss_curves(self, r: dict):
        train_h = r.get("train_history", [])
        val_h   = r.get("val_history", [])

        self.ax_loss.clear()
        epochs = list(range(1, len(train_h) + 1))
        if train_h:
            self.ax_loss.plot(epochs, train_h, label="Train",
                              linewidth=1.5, color="#1976D2")
        val_valid = [(i + 1, v) for i, v in enumerate(val_h) if v is not None]
        if val_valid:
            ex, ey = zip(*val_valid)
            self.ax_loss.plot(list(ex), list(ey), label="Val",
                              linewidth=1.5, color="#F57C00")
        best_ep = r.get("best_epoch")
        if best_ep and best_ep <= len(train_h):
            self.ax_loss.axvline(best_ep, color="green", linestyle="--", alpha=0.7,
                                 label=f"Best ep={best_ep}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title(f"Loss — {r.get('run_id', '')}")
        handles, _ = self.ax_loss.get_legend_handles_labels()
        if handles:
            self.ax_loss.legend(fontsize=8)
        self.ax_loss.grid(True, alpha=0.3)
        self.canvas_loss.draw()

    # ------------------------------------------------------------------ confusion matrix
    def _draw_cm_figure(self, fig, canvas, r):
        """Draw confusion matrix onto fig (assumed freshly clf'd with one subplot).
        Returns (cm_array, class_names, total) on success, or None on failure/N/A."""
        ax = fig.axes[0]

        if r.get("mode") != "classification":
            ax.text(
                0.5, 0.5, "Regression run\n(no confusion matrix)",
                ha="center", va="center", transform=ax.transAxes,
                color="#aaa", fontsize=11,
            )
            canvas.draw()
            return None

        tm = r.get("final_test_metrics", {})
        vm = r.get("final_val_metrics",  {})
        preds  = tm.get("predictions") or vm.get("predictions")
        labels = tm.get("true_labels")  or vm.get("true_labels")
        source = "Test (held out)" if tm.get("predictions") else "Validation (monitored)"
        class_names = r.get("class_names", [])

        if not preds or not labels:
            ax.text(
                0.5, 0.5, "No prediction data stored\n(re-train to generate)",
                ha="center", va="center", transform=ax.transAxes,
                color="#aaa", fontsize=11,
            )
            canvas.draw()
            return None

        p = np.array(preds,  dtype=int)
        y = np.array(labels, dtype=int)
        n = len(class_names) if class_names else (max(max(p), max(y)) + 1)
        names = class_names if class_names else [str(i) for i in range(n)]

        cm = np.zeros((n, n), dtype=int)
        for pred_i, true_i in zip(p, y):
            if 0 <= true_i < n and 0 <= pred_i < n:
                cm[true_i, pred_i] += 1

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(
            f"Confusion Matrix ({source}) — {r.get('run_id', '')}",
            fontsize=9, fontweight="bold",
        )
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] > vmax * 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        canvas.draw()
        return cm, names, len(p)

    def _draw_confusion_matrix(self, r: dict):
        # ── Main Confusion Matrix tab ─────────────────────────────────────
        self.fig_cm.clf()
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")

        result = self._draw_cm_figure(self.fig_cm, self.canvas_cm, r)
        if result is not None:
            cm, names, total = result
            self._fill_cls_metrics_table(
                self.tbl_cls_metrics, self.lbl_overall_metrics, cm, names, total
            )

        # ── Overview tab (mirror) ─────────────────────────────────────────
        self.fig_cm_combo.clf()
        self.ax_cm_combo = self.fig_cm_combo.add_subplot(111)
        self.tbl_cls_metrics_combo.setRowCount(0)
        self.lbl_overall_metrics_combo.setText("")

        result2 = self._draw_cm_figure(self.fig_cm_combo, self.canvas_cm_combo, r)
        if result2 is not None:
            cm2, names2, total2 = result2
            self._fill_cls_metrics_table(
                self.tbl_cls_metrics_combo, self.lbl_overall_metrics_combo,
                cm2, names2, total2
            )

    def _fill_cls_metrics_table(self, tbl: QTableWidget, lbl: QLabel,
                                 cm: np.ndarray, names: list, total: int):
        n = len(names)
        tbl.setRowCount(n)
        macro_f1_sum    = 0.0
        weighted_f1_sum = 0.0

        for i, name in enumerate(names):
            tp      = cm[i, i]
            fp      = cm[:, i].sum() - tp
            fn      = cm[i, :].sum() - tp
            support = int(cm[i, :].sum())
            prec    = tp / (tp + fp + 1e-9)
            rec     = tp / (tp + fn + 1e-9)
            f1      = 2 * prec * rec / (prec + rec + 1e-9)
            macro_f1_sum    += f1
            weighted_f1_sum += f1 * support

            row_vals = [name, str(support),
                        f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"]
            for col, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4:                    # colour-code F1
                    fv = float(val)
                    if fv >= 0.7:
                        item.setBackground(QColor("#c8e6c9"))
                    elif fv >= 0.4:
                        item.setBackground(QColor("#fff9c4"))
                    else:
                        item.setBackground(QColor("#ffcdd2"))
                tbl.setItem(i, col, item)

        correct  = int(np.diag(cm).sum())
        accuracy = correct / max(total, 1)
        macro_f1    = macro_f1_sum    / n
        weighted_f1 = weighted_f1_sum / max(total, 1)
        lbl.setText(
            f"Overall  accuracy={accuracy:.3f}  |  "
            f"macro-F1={macro_f1:.3f}  |  "
            f"weighted-F1={weighted_f1:.3f}  |  "
            f"n={total}"
        )

    # ------------------------------------------------------------------ performance eval
    def _fill_perf_widgets(self, r: dict,
                           grp_thesis, lbl_acc, lbl_result,
                           grp_metrics, tbl_perf):
        """Populate thesis-goal + metrics table widgets for one result dict."""
        mode = r.get("mode", "regression")
        tm   = r.get("final_test_metrics", {})
        vm   = r.get("final_val_metrics",  {})

        if mode == "classification":
            preds  = tm.get("predictions") or vm.get("predictions")
            labels = tm.get("true_labels")  or vm.get("true_labels")
            class_names = r.get("class_names", [])
            source = "Test (held out)" if tm.get("predictions") else "Validation (monitored)"

            if not preds or not labels:
                grp_thesis.setVisible(False)
                tbl_perf.setRowCount(0)
                grp_metrics.setTitle("Performance Evaluation (no data — re-train)")
                return

            n = len(class_names) if class_names else (max(max(preds), max(labels)) + 1)
            m = _compute_cls_perf_metrics(preds, labels, n)
            acc = m["accuracy"]

            grp_thesis.setVisible(True)
            grp_thesis.setTitle(
                f"Thesis Goal  —  ≥ 70% correct prediction  ({source})"
            )
            pct = acc * 100
            lbl_acc.setText(f"{pct:.1f}%")

            if acc >= 0.70:
                color      = "#1b5e20"
                result_txt = "✓  Meets the ≥ 70% accuracy target"
            elif acc >= 0.60:
                color      = "#e65100"
                gap        = (0.70 - acc) * 100
                result_txt = f"⚠  {gap:.1f} pp below target — close"
            else:
                color      = "#b71c1c"
                gap        = (0.70 - acc) * 100
                result_txt = f"✗  {gap:.1f} pp below the 70% target"

            lbl_acc.setStyleSheet(
                f"color:{color}; font-size:{lbl_acc.font().pointSize()}px; font-weight:bold;"
            )
            lbl_result.setText(result_txt)
            lbl_result.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{color};"
            )

            grp_metrics.setTitle(f"Classification Metrics  ({source})")
            rows = [
                ("Accuracy",          m["accuracy"],          self._rate_accuracy(m["accuracy"])),
                ("Balanced Accuracy", m["balanced_accuracy"],  self._rate_accuracy(m["balanced_accuracy"])),
                ("Cohen's Kappa",     m["kappa"],              self._rate_kappa(m["kappa"])),
                ("MCC",               m["mcc"],                self._rate_mcc(m["mcc"])),
                ("Macro F1",          m["macro_f1"],           self._rate_f1(m["macro_f1"])),
                ("Weighted F1",       m["weighted_f1"],        self._rate_f1(m["weighted_f1"])),
            ]
            tbl_perf.setRowCount(len(rows))
            n_cols = tbl_perf.columnCount()
            for i, (name, val, (rating, bg)) in enumerate(rows):
                cells = [
                    QTableWidgetItem(name),
                    QTableWidgetItem(f"{val:.4f}"),
                    QTableWidgetItem(rating),
                    QTableWidgetItem(_PERF_BRIEF.get(name, "")),
                ]
                for col in range(min(n_cols, len(cells))):
                    cell = cells[col]
                    cell.setTextAlignment(
                        Qt.AlignCenter if col < 3 else Qt.AlignLeft | Qt.AlignVCenter
                    )
                    if bg and col < 3:
                        cell.setBackground(QColor(bg))
                    tbl_perf.setItem(i, col, cell)

        else:
            # ── Regression mode ───────────────────────────────────────────
            grp_thesis.setVisible(False)
            m_src  = tm if tm else vm
            source = "Test (held out)" if tm else "Validation (monitored)"
            grp_metrics.setTitle(f"Regression Metrics  ({source})")

            _reg_brief = {
                "RMSE": "Root Mean Squared Error — avg prediction error in Pb% units (lower = better)",
                "MAE":  "Mean Absolute Error — avg absolute error in Pb% units (lower = better)",
                "R²":   "Coefficient of determination: 1=perfect, 0=mean baseline, <0=worse than mean",
            }
            rows_reg = []
            for key, name in [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")]:
                v = m_src.get(key)
                if v is not None:
                    rate = (self._rate_rmse(v) if key == "rmse" else
                            self._rate_r2(v)   if key == "r2"   else ("—", None))
                    rows_reg.append((name, v, rate))

            tbl_perf.setRowCount(len(rows_reg))
            n_cols = tbl_perf.columnCount()
            for i, (name, val, (rating, bg)) in enumerate(rows_reg):
                cells = [
                    QTableWidgetItem(name),
                    QTableWidgetItem(f"{val:.4f}"),
                    QTableWidgetItem(rating),
                    QTableWidgetItem(_reg_brief.get(name, "")),
                ]
                for col in range(min(n_cols, len(cells))):
                    cell = cells[col]
                    cell.setTextAlignment(
                        Qt.AlignCenter if col < 3 else Qt.AlignLeft | Qt.AlignVCenter
                    )
                    if bg and col < 3:
                        cell.setBackground(QColor(bg))
                    tbl_perf.setItem(i, col, cell)

    def _draw_performance_eval(self, r: dict):
        """Fill the Performance Evaluation tab and the Overview tab for the selected run."""
        # Main Performance Evaluation tab
        self._fill_perf_widgets(
            r,
            self.grp_thesis, self.lbl_thesis_acc, self.lbl_thesis_result,
            self.grp_metrics_summary, self.tbl_perf,
        )
        # Ensure big-number font is correct for main tab (46px)
        self.lbl_thesis_acc.setFont(QFont("Arial", 46, QFont.Bold))

        # Overview tab (mirror)
        self._fill_perf_widgets(
            r,
            self.grp_thesis_combo, self.lbl_thesis_acc_combo, self.lbl_thesis_result_combo,
            self.grp_metrics_combo, self.tbl_perf_combo,
        )
        # Smaller font for the compact combo header
        self.lbl_thesis_acc_combo.setFont(QFont("Arial", 36, QFont.Bold))

    def _perf_clear(self, msg: str = ""):
        self.grp_thesis.setVisible(False)
        self.tbl_perf.setRowCount(0)
        self.grp_metrics_summary.setTitle(
            f"Performance Evaluation  ({msg})" if msg else "Performance Evaluation"
        )
        # Also clear Overview perf section
        self.grp_thesis_combo.setVisible(False)
        self.tbl_perf_combo.setRowCount(0)
        self.grp_metrics_combo.setTitle("Classification Metrics")

    # ── Rating helpers ────────────────────────────────────────────────────
    @staticmethod
    def _rate_accuracy(v: float):
        if v >= 0.70: return ("✓ Good",     "#c8e6c9")
        if v >= 0.55: return ("~ Moderate", "#fff9c4")
        return             ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_kappa(v: float):
        if v >= 0.81: return ("✓ Almost perfect", "#c8e6c9")
        if v >= 0.61: return ("✓ Substantial",    "#c8e6c9")
        if v >= 0.41: return ("~ Moderate",       "#fff9c4")
        if v >= 0.21: return ("~ Fair",           "#fff9c4")
        return             ("✗ Slight / None",   "#ffcdd2")

    @staticmethod
    def _rate_mcc(v: float):
        if v >= 0.70: return ("✓ Strong",   "#c8e6c9")
        if v >= 0.50: return ("✓ Good",     "#c8e6c9")
        if v >= 0.30: return ("~ Fair",     "#fff9c4")
        return             ("✗ Weak",      "#ffcdd2")

    @staticmethod
    def _rate_f1(v: float):
        if v >= 0.70: return ("✓ Good",     "#c8e6c9")
        if v >= 0.50: return ("~ Moderate", "#fff9c4")
        return             ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_rmse(v: float):
        # Lower is better; thresholds in Pb% units
        if v <= 2.0: return ("✓ Good",     "#c8e6c9")
        if v <= 5.0: return ("~ Moderate", "#fff9c4")
        return            ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_r2(v: float):
        if v >= 0.80: return ("✓ Strong",   "#c8e6c9")
        if v >= 0.60: return ("✓ Good",     "#c8e6c9")
        if v >= 0.40: return ("~ Moderate", "#fff9c4")
        return             ("✗ Weak",      "#ffcdd2")

    # ------------------------------------------------------------------ placeholder
    def _draw_placeholder(self):
        # Loss curve
        self.ax_loss.clear()
        self.ax_loss.set_title("Loss Curves")
        self.ax_loss.text(0.5, 0.5, "Select a run", ha="center", va="center",
                          transform=self.ax_loss.transAxes, color="#aaa", fontsize=11)
        self.canvas_loss.draw()

        # Confusion matrix (main tab)
        self.fig_cm.clf()
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.ax_cm.set_title("Confusion Matrix")
        self.ax_cm.text(0.5, 0.5, "Select a run", ha="center", va="center",
                        transform=self.ax_cm.transAxes, color="#aaa", fontsize=11)
        self.canvas_cm.draw()
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")

        # Confusion matrix (Overview tab)
        self.fig_cm_combo.clf()
        self.ax_cm_combo = self.fig_cm_combo.add_subplot(111)
        self.ax_cm_combo.set_title("Confusion Matrix")
        self.ax_cm_combo.text(0.5, 0.5, "Select a run", ha="center", va="center",
                              transform=self.ax_cm_combo.transAxes, color="#aaa", fontsize=11)
        self.canvas_cm_combo.draw()
        self.tbl_cls_metrics_combo.setRowCount(0)
        self.lbl_overall_metrics_combo.setText("")

        # Performance evaluation
        self._perf_clear()
