"""Results panel: table of all runs + loss curves + performance evaluation."""
import os
import csv
import math

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QSplitter, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QTabWidget, QSizePolicy,
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
    "Val Loss", "Val RMSE", "Val MAE", "Val R²", "Val Acc", "Val F1",
    "Test RMSE", "Test MAE", "Test R²", "Test Acc", "Test F1",
    "Time (s)", "Checkpoint",
]

_CM_HEADERS = [
    ("Class",     "The class name."),
    ("Support",   "Number of actual images of this class in the evaluation set."),
    ("Precision", "TP / (TP + FP)  —  of all predicted as this class, fraction correct."),
    ("Recall",    "TP / (TP + FN)  —  of all actual of this class, fraction found."),
    ("F1",        "Harmonic mean of Precision and Recall.\nGreen ≥ 0.70 | Yellow ≥ 0.40 | Red < 0.40"),
]

_PERF_HDRS = [
    ("Metric",           "Name of the evaluation metric"),
    ("Value",            "Computed value for this run"),
    ("Rating",           "Qualitative interpretation"),
    ("What it measures", "One-line description"),
]

_PERF_BRIEF = {
    "Accuracy":          "Correct predictions / total images",
    "Balanced Accuracy": "Mean recall per class — robust to class imbalance",
    "Cohen's Kappa":     "Agreement corrected for chance  (0=random, 1=perfect)",
    "MCC":               "Correlation coeff. for multi-class  (−1 to +1)",
    "Macro F1":          "Mean F1 per class — equal weight to all classes",
    "Weighted F1":       "Mean F1 per class — weighted by class sample count",
}


def _compute_cls_perf_metrics(preds, labels, n_classes):
    p = np.array(preds,  dtype=int)
    y = np.array(labels, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred_i, true_i in zip(p, y):
        if 0 <= true_i < n_classes and 0 <= pred_i < n_classes:
            cm[true_i, pred_i] += 1
    total    = len(p)
    correct  = int(np.diag(cm).sum())
    accuracy = correct / max(total, 1)
    per_cls_recall = [float(cm[i, i]) / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
                      for i in range(n_classes)]
    balanced_acc = float(np.mean(per_cls_recall))
    po = accuracy
    pe = sum((int(cm[i, :].sum()) * int(cm[:, i].sum())) for i in range(n_classes)) \
         / (total ** 2) if total > 0 else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-9 else 0.0
    t_k = np.sum(cm, axis=1).astype(float)
    p_k = np.sum(cm, axis=0).astype(float)
    c   = float(np.diag(cm).sum())
    s   = float(cm.sum())
    mcc_num = c * s - float(np.dot(t_k, p_k))
    mcc_den = np.sqrt((s**2 - float(np.dot(p_k, p_k))) * (s**2 - float(np.dot(t_k, t_k))))
    mcc = mcc_num / mcc_den if mcc_den > 1e-9 else 0.0
    macro_f1_sum = weighted_f1_sum = 0.0
    for i in range(n_classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        macro_f1_sum += f1; weighted_f1_sum += f1 * support
    return {
        "accuracy": accuracy, "balanced_accuracy": balanced_acc,
        "kappa": kappa, "mcc": float(mcc),
        "macro_f1": macro_f1_sum / n_classes if n_classes else 0.0,
        "weighted_f1": weighted_f1_sum / max(total, 1),
        "cm": cm,
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

        self.chart_tabs = QTabWidget()
        splitter.addWidget(self.chart_tabs)

        # ══════════════════════════════════════════════════════════════════
        # Tab 0 — Performance Evaluation
        # Layout:
        #   Top strip  : Thesis goal  |  Summary metrics table
        #   Bottom area: Confusion matrix  |  Per-class table
        # ══════════════════════════════════════════════════════════════════
        perf_w = QWidget()
        perf_v = QVBoxLayout(perf_w)
        perf_v.setContentsMargins(4, 4, 4, 4)
        perf_v.setSpacing(4)

        # ── Top strip ─────────────────────────────────────────────────────
        top_split = QSplitter(Qt.Horizontal)
        top_split.setMaximumHeight(170)

        # Left of top: thesis goal (big accuracy number)
        self.grp_thesis = QGroupBox("Accuracy  (≥ 70% target)")
        thesis_h = QHBoxLayout(self.grp_thesis)
        thesis_h.setSpacing(12)
        self.lbl_thesis_acc = QLabel("—")
        self.lbl_thesis_acc.setFont(QFont("Arial", 40, QFont.Bold))
        self.lbl_thesis_acc.setAlignment(Qt.AlignCenter)
        self.lbl_thesis_acc.setStyleSheet("color:#555;")
        self.lbl_thesis_acc.setMinimumWidth(120)
        thesis_h.addWidget(self.lbl_thesis_acc)
        self.lbl_thesis_result = QLabel("Select a run to evaluate.")
        self.lbl_thesis_result.setStyleSheet("font-size:13px; font-weight:bold; color:#555;")
        self.lbl_thesis_result.setWordWrap(True)
        thesis_h.addWidget(self.lbl_thesis_result, 1)
        top_split.addWidget(self.grp_thesis)

        # Right of top: summary metrics table
        self.grp_metrics_summary = QGroupBox("Classification Metrics")
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
        summary_v.addWidget(self.tbl_perf)
        top_split.addWidget(self.grp_metrics_summary)
        top_split.setSizes([280, 520])
        perf_v.addWidget(top_split)

        # ── Bottom area ───────────────────────────────────────────────────
        bottom_split = QSplitter(Qt.Horizontal)

        # Left: confusion matrix figure
        cm_w = QWidget()
        cm_v = QVBoxLayout(cm_w)
        cm_v.setContentsMargins(0, 0, 0, 0)
        self.fig_cm    = Figure(figsize=(4, 4), tight_layout=True)
        self.ax_cm     = self.fig_cm.add_subplot(111)
        self.canvas_cm = FigureCanvas(self.fig_cm)
        cm_v.addWidget(self.canvas_cm)
        bottom_split.addWidget(cm_w)

        # Right: per-class metrics table
        cls_w = QWidget()
        cls_v = QVBoxLayout(cls_w)
        cls_v.setContentsMargins(6, 2, 4, 4)
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
        cls_v.addWidget(self.tbl_cls_metrics, 1)

        self.lbl_overall_metrics = QLabel("")
        self.lbl_overall_metrics.setStyleSheet("font-size:10px; color:#333; margin-top:2px;")
        self.lbl_overall_metrics.setWordWrap(True)
        cls_v.addWidget(self.lbl_overall_metrics)

        bottom_split.addWidget(cls_w)
        bottom_split.setSizes([430, 320])
        perf_v.addWidget(bottom_split, 1)   # stretch=1 → fills remaining height

        self.chart_tabs.addTab(perf_w, "Performance Evaluation")

        # ══════════════════════════════════════════════════════════════════
        # Tab 1 — Loss Curves
        # ══════════════════════════════════════════════════════════════════
        loss_w = QWidget()
        loss_v = QVBoxLayout(loss_w)
        loss_v.setContentsMargins(2, 2, 2, 2)
        self.fig_loss    = Figure(figsize=(6, 3), tight_layout=True)
        self.ax_loss     = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        loss_v.addWidget(self.canvas_loss)
        self.chart_tabs.addTab(loss_w, "Loss Curves")

        # ══════════════════════════════════════════════════════════════════
        # Tab 2 — Metric Glossary
        # ══════════════════════════════════════════════════════════════════
        glossary_w     = QWidget()
        glossary_outer = QVBoxLayout(glossary_w)
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
            "Overall % of images classified correctly.  "
            "= (correctly classified) / (total images).  "
            "Primary target: ≥ 70%.<br><br>"
            "<b style='font-size:13px'>Balanced Accuracy</b><br>"
            "Average recall per class, each class weighted equally — "
            "not fooled by class imbalance.<br><br>"
            "<b style='font-size:13px'>Cohen's Kappa (κ)</b><br>"
            "Better-than-chance agreement.  κ=0 → random; κ=1 → perfect.  "
            "Bands: ≥0.81 Almost perfect | 0.61–0.80 Substantial | "
            "0.41–0.60 Moderate | 0.21–0.40 Fair | ≤0.20 Slight.<br><br>"
            "<b style='font-size:13px'>Matthews Correlation Coefficient (MCC)</b><br>"
            "Multi-class correlation: −1 (all wrong) → 0 (random) → +1 (perfect).  "
            "Best single number for imbalanced classification.<br><br>"
            "<b style='font-size:13px'>Macro F1</b><br>"
            "F1 averaged per class with equal weight.  "
            "Penalises poor performance on any class.<br><br>"
            "<b style='font-size:13px'>Weighted F1</b><br>"
            "F1 per class weighted by sample count — realistic overall picture.<br><br>"
            "<b style='font-size:13px'>Precision (per class)</b><br>"
            "TP / (TP+FP).  Low → many false alarms for this class.<br><br>"
            "<b style='font-size:13px'>Recall (per class)</b><br>"
            "TP / (TP+FN).  Low → model frequently misses this class.<br><br>"
            "<b style='font-size:13px'>F1 (per class)</b><br>"
            "2×P×R / (P+R).  Low if either Precision or Recall is low.<br><br>"
            "<b style='font-size:13px'>Support</b><br>"
            "Actual images of this class in the eval set.  "
            "Very low support → unreliable per-class metrics.<br><br>"
            "<b style='font-size:13px'>RMSE</b>  (regression only)<br>"
            "Root Mean Squared Error in Pb% units.  Lower is better.<br><br>"
            "<b style='font-size:13px'>MAE</b>  (regression only)<br>"
            "Mean Absolute Error in Pb% units.  Lower is better.<br><br>"
            "<b style='font-size:13px'>R²</b>  (regression only)<br>"
            "1=perfect, 0=mean baseline, negative=worse than mean.  Higher is better."
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
        save_result(r, self._output_dir or "./results")

    def _load_previous(self):
        from ..utils.results_saver import load_results
        path = QFileDialog.getExistingDirectory(
            self, "Select Results Directory", self._output_dir
        )
        if not path:
            return
        loaded = load_results(path)
        if not loaded:
            QMessageBox.information(self, "No Results",
                                    "No result.json files found in that directory.")
            return
        existing_ids = {r.get("run_id") for r in self._results}
        new_count = 0
        for r in loaded:
            if r.get("run_id") not in existing_ids:
                self._results.append(r)
                self._append_row(r)
                new_count += 1
        self._update_best_label()
        QMessageBox.information(self, "Loaded",
                                f"Loaded {new_count} new run(s) from:\n{path}")

    # ------------------------------------------------------------------ table
    def _append_row(self, r: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        params = r.get("params", {})
        vm = r.get("final_val_metrics",  {})
        tm = r.get("final_test_metrics", {})

        def _v(m, key, fmt="{:.4f}"):
            v = m.get(key)
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
            f"{best_val:.4f}"   if best_val  is not None else "—",
            _v(vm, "rmse"), _v(vm, "mae"), _v(vm, "r2"),
            _v(vm, "accuracy", "{:.3f}"), _v(vm, "f1", "{:.3f}"),
            _v(tm, "rmse"), _v(tm, "mae"), _v(tm, "r2"),
            _v(tm, "accuracy", "{:.3f}"), _v(tm, "f1", "{:.3f}"),
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
            self.lbl_best.setText(f"Best: {best.get('run_id', '')}  {metric_str}")
        else:
            self.lbl_best.setText("")
        for row in range(self.table.rowCount()):
            is_best = (best and self.table.item(row, 0) and
                       self.table.item(row, 0).text() == best.get("run_id", ""))
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
        def _m(r):
            return r.get("final_test_metrics") or r.get("final_val_metrics", {})
        if valid[0].get("mode") == "classification":
            return max(valid, key=lambda r: _m(r).get("accuracy", 0.0))
        return min(valid, key=lambda r: _m(r).get("rmse", math.inf))

    def _clear_all(self):
        if QMessageBox.question(self, "Clear", "Clear all results?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
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
        self.ax_loss.clear()
        train_h = r.get("train_history", [])
        val_h   = r.get("val_history", [])
        epochs  = list(range(1, len(train_h) + 1))
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
            self.ax_loss.axvline(best_ep, color="green", linestyle="--",
                                 alpha=0.7, label=f"Best ep={best_ep}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title(f"Loss — {r.get('run_id', '')}")
        if self.ax_loss.get_legend_handles_labels()[0]:
            self.ax_loss.legend(fontsize=8)
        self.ax_loss.grid(True, alpha=0.3)
        self.canvas_loss.draw()

    # ------------------------------------------------------------------ confusion matrix
    def _draw_confusion_matrix(self, r: dict):
        self.fig_cm.clf()
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")

        ax = self.ax_cm
        if r.get("mode") != "classification":
            ax.text(0.5, 0.5, "Regression run\n(no confusion matrix)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#aaa", fontsize=11)
            self.canvas_cm.draw()
            return

        tm = r.get("final_test_metrics", {})
        vm = r.get("final_val_metrics",  {})
        preds  = tm.get("predictions") or vm.get("predictions")
        labels = tm.get("true_labels")  or vm.get("true_labels")
        source = "Test (held out)" if tm.get("predictions") else "Validation (monitored)"
        class_names = r.get("class_names", [])

        if not preds or not labels:
            ax.text(0.5, 0.5, "No prediction data stored\n(re-train to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#aaa", fontsize=11)
            self.canvas_cm.draw()
            return

        p = np.array(preds, dtype=int)
        y = np.array(labels, dtype=int)
        n = len(class_names) if class_names else (max(max(p), max(y)) + 1)
        names = class_names if class_names else [str(i) for i in range(n)]
        cm = np.zeros((n, n), dtype=int)
        for pred_i, true_i in zip(p, y):
            if 0 <= true_i < n and 0 <= pred_i < n:
                cm[true_i, pred_i] += 1

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(f"Confusion Matrix ({source})\n{r.get('run_id', '')}",
                     fontsize=9, fontweight="bold")
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i, j] > vmax * 0.6 else "black")
        self.fig_cm.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()
        self._fill_cls_metrics_table(cm, names, len(p))

    def _fill_cls_metrics_table(self, cm: np.ndarray, names: list, total: int):
        n = len(names)
        self.tbl_cls_metrics.setRowCount(n)
        macro_f1_sum = weighted_f1_sum = 0.0
        for i, name in enumerate(names):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
            support = int(cm[i, :].sum())
            prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            macro_f1_sum += f1; weighted_f1_sum += f1 * support
            for col, val in enumerate([name, str(support),
                                       f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    fv = float(val)
                    item.setBackground(QColor(
                        "#c8e6c9" if fv >= 0.7 else "#fff9c4" if fv >= 0.4 else "#ffcdd2"
                    ))
                self.tbl_cls_metrics.setItem(i, col, item)

        acc  = int(np.diag(cm).sum()) / max(total, 1)
        mf1  = macro_f1_sum / n
        wf1  = weighted_f1_sum / max(total, 1)
        self.lbl_overall_metrics.setText(
            f"accuracy={acc:.3f}  |  macro-F1={mf1:.3f}  |  "
            f"weighted-F1={wf1:.3f}  |  n={total}"
        )

    # ------------------------------------------------------------------ performance eval
    def _draw_performance_eval(self, r: dict):
        mode = r.get("mode", "regression")
        tm   = r.get("final_test_metrics", {})
        vm   = r.get("final_val_metrics",  {})

        if mode == "classification":
            preds  = tm.get("predictions") or vm.get("predictions")
            labels = tm.get("true_labels")  or vm.get("true_labels")
            class_names = r.get("class_names", [])
            source = "Test (held out)" if tm.get("predictions") else "Validation (monitored)"

            if not preds or not labels:
                self._perf_clear("no data — re-train to generate metrics.")
                return

            n = len(class_names) if class_names else (max(max(preds), max(labels)) + 1)
            m = _compute_cls_perf_metrics(preds, labels, n)
            acc = m["accuracy"]

            self.grp_thesis.setVisible(True)
            self.grp_thesis.setTitle(
                f"Thesis Goal  —  ≥ 70% correct prediction  ({source})"
            )
            self.lbl_thesis_acc.setText(f"{acc * 100:.1f}%")

            if acc >= 0.70:
                color = "#1b5e20"; txt = "✓  Meets the ≥ 70% accuracy target"
            elif acc >= 0.60:
                color = "#e65100"; txt = f"⚠  {(0.70-acc)*100:.1f} pp below target — close"
            else:
                color = "#b71c1c"; txt = f"✗  {(0.70-acc)*100:.1f} pp below the 70% target"

            self.lbl_thesis_acc.setStyleSheet(
                f"color:{color}; font-size:40px; font-weight:bold;"
            )
            self.lbl_thesis_result.setText(txt)
            self.lbl_thesis_result.setStyleSheet(
                f"font-size:13px; font-weight:bold; color:{color};"
            )

            self.grp_metrics_summary.setTitle(f"Classification Metrics  ({source})")
            rows = [
                ("Accuracy",          m["accuracy"],          self._rate_accuracy(m["accuracy"])),
                ("Balanced Accuracy", m["balanced_accuracy"],  self._rate_accuracy(m["balanced_accuracy"])),
                ("Cohen's Kappa",     m["kappa"],              self._rate_kappa(m["kappa"])),
                ("MCC",               m["mcc"],                self._rate_mcc(m["mcc"])),
                ("Macro F1",          m["macro_f1"],           self._rate_f1(m["macro_f1"])),
                ("Weighted F1",       m["weighted_f1"],        self._rate_f1(m["weighted_f1"])),
            ]
            self.tbl_perf.setRowCount(len(rows))
            for i, (name, val, (rating, bg)) in enumerate(rows):
                cells = [QTableWidgetItem(name), QTableWidgetItem(f"{val:.4f}"),
                         QTableWidgetItem(rating),
                         QTableWidgetItem(_PERF_BRIEF.get(name, ""))]
                for col, cell in enumerate(cells):
                    cell.setTextAlignment(
                        Qt.AlignCenter if col < 3 else Qt.AlignLeft | Qt.AlignVCenter
                    )
                    if bg and col < 3:
                        cell.setBackground(QColor(bg))
                    self.tbl_perf.setItem(i, col, cell)

        else:
            self.grp_thesis.setVisible(False)
            m_src  = tm if tm else vm
            source = "Test (held out)" if tm else "Validation (monitored)"
            self.grp_metrics_summary.setTitle(f"Regression Metrics  ({source})")
            _reg = {
                "RMSE": "Root Mean Squared Error — avg prediction error in Pb% (lower = better)",
                "MAE":  "Mean Absolute Error — avg absolute error in Pb% (lower = better)",
                "R²":   "1=perfect, 0=mean baseline, <0=worse than mean",
            }
            rows_r = [(n, m_src[k], self._rate_rmse(m_src[k]) if k=="rmse" else
                       self._rate_r2(m_src[k]) if k=="r2" else ("—", None))
                      for k, n in [("rmse","RMSE"),("mae","MAE"),("r2","R²")]
                      if m_src.get(k) is not None]
            self.tbl_perf.setRowCount(len(rows_r))
            for i, (name, val, (rating, bg)) in enumerate(rows_r):
                cells = [QTableWidgetItem(name), QTableWidgetItem(f"{val:.4f}"),
                         QTableWidgetItem(rating), QTableWidgetItem(_reg.get(name, ""))]
                for col, cell in enumerate(cells):
                    cell.setTextAlignment(
                        Qt.AlignCenter if col < 3 else Qt.AlignLeft | Qt.AlignVCenter
                    )
                    if bg and col < 3:
                        cell.setBackground(QColor(bg))
                    self.tbl_perf.setItem(i, col, cell)

    def _perf_clear(self, msg: str = ""):
        self.grp_thesis.setVisible(False)
        self.tbl_perf.setRowCount(0)
        self.grp_metrics_summary.setTitle(
            f"Performance Evaluation  ({msg})" if msg else "Performance Evaluation"
        )

    # ── Rating helpers ────────────────────────────────────────────────────
    @staticmethod
    def _rate_accuracy(v):
        if v >= 0.70: return ("✓ Good",     "#c8e6c9")
        if v >= 0.55: return ("~ Moderate", "#fff9c4")
        return             ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_kappa(v):
        if v >= 0.81: return ("✓ Almost perfect", "#c8e6c9")
        if v >= 0.61: return ("✓ Substantial",    "#c8e6c9")
        if v >= 0.41: return ("~ Moderate",       "#fff9c4")
        if v >= 0.21: return ("~ Fair",           "#fff9c4")
        return             ("✗ Slight / None",   "#ffcdd2")

    @staticmethod
    def _rate_mcc(v):
        if v >= 0.70: return ("✓ Strong", "#c8e6c9")
        if v >= 0.50: return ("✓ Good",   "#c8e6c9")
        if v >= 0.30: return ("~ Fair",   "#fff9c4")
        return             ("✗ Weak",    "#ffcdd2")

    @staticmethod
    def _rate_f1(v):
        if v >= 0.70: return ("✓ Good",     "#c8e6c9")
        if v >= 0.50: return ("~ Moderate", "#fff9c4")
        return             ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_rmse(v):
        if v <= 2.0: return ("✓ Good",     "#c8e6c9")
        if v <= 5.0: return ("~ Moderate", "#fff9c4")
        return            ("✗ Poor",      "#ffcdd2")

    @staticmethod
    def _rate_r2(v):
        if v >= 0.80: return ("✓ Strong",   "#c8e6c9")
        if v >= 0.60: return ("✓ Good",     "#c8e6c9")
        if v >= 0.40: return ("~ Moderate", "#fff9c4")
        return             ("✗ Weak",      "#ffcdd2")

    # ------------------------------------------------------------------ placeholder
    def _draw_placeholder(self):
        self.ax_loss.clear()
        self.ax_loss.set_title("Loss Curves")
        self.ax_loss.text(0.5, 0.5, "Select a run", ha="center", va="center",
                          transform=self.ax_loss.transAxes, color="#aaa", fontsize=11)
        self.canvas_loss.draw()

        self.fig_cm.clf()
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.ax_cm.set_title("Confusion Matrix")
        self.ax_cm.text(0.5, 0.5, "Select a run", ha="center", va="center",
                        transform=self.ax_cm.transAxes, color="#aaa", fontsize=11)
        self.canvas_cm.draw()
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")

        self._perf_clear()
