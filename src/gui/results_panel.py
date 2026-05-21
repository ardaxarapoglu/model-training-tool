"""Results panel: table of all runs + loss curves + metrics + confusion matrix."""
import os
import csv
import math

import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QSplitter, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QTabWidget,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


RESULT_COLS = [
    "Run ID", "Mode", "Batch", "LR", "Optimizer", "WD", "Loss Fn",
    "Best Ep.", "Train Loss",
    "Test RMSE", "Test MAE", "Test R²", "Test Acc", "Test F1",
    "Val RMSE",  "Val MAE",  "Val R²",  "Val Acc",  "Val F1",
    "Time (s)", "Checkpoint",
]


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._results = []
        self._output_dir = "./results"
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_load = QPushButton("Load Previous Runs…")
        self.btn_load.setToolTip("Load saved result.json files from a results directory")
        self.btn_load.clicked.connect(self._load_previous)
        self.lbl_best = QLabel("")
        self.lbl_best.setStyleSheet("font-weight:bold;color:#1b5e20;")
        toolbar.addWidget(self.btn_clear)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_load)
        toolbar.addWidget(self.lbl_best)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter)

        # ---- Table ----
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

        # ---- Charts tab widget ----
        self.chart_tabs = QTabWidget()
        splitter.addWidget(self.chart_tabs)

        # Tab 1 — Loss curves
        loss_w = QWidget()
        loss_layout = QVBoxLayout(loss_w)
        loss_layout.setContentsMargins(2, 2, 2, 2)
        self.fig_loss = Figure(figsize=(6, 3), tight_layout=True)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        loss_layout.addWidget(self.canvas_loss)
        self.chart_tabs.addTab(loss_w, "Loss Curves")

        # Tab 2 — Final metrics bar chart
        metrics_w = QWidget()
        metrics_layout = QVBoxLayout(metrics_w)
        metrics_layout.setContentsMargins(2, 2, 2, 2)
        self.fig_val = Figure(figsize=(4, 3), tight_layout=True)
        self.ax_val = self.fig_val.add_subplot(111)
        self.canvas_val = FigureCanvas(self.fig_val)
        metrics_layout.addWidget(self.canvas_val)
        self.chart_tabs.addTab(metrics_w, "Metrics")

        # Tab 3 — Confusion matrix + per-class table
        cm_w = QWidget()
        cm_split = QSplitter(Qt.Horizontal)
        cm_outer = QVBoxLayout(cm_w)
        cm_outer.setContentsMargins(2, 2, 2, 2)
        cm_outer.addWidget(cm_split)

        # Left: confusion matrix plot
        cm_plot_w = QWidget()
        cm_plot_v = QVBoxLayout(cm_plot_w)
        cm_plot_v.setContentsMargins(0, 0, 0, 0)
        self.fig_cm = Figure(figsize=(4, 4), tight_layout=True)
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.canvas_cm = FigureCanvas(self.fig_cm)
        cm_plot_v.addWidget(self.canvas_cm)
        cm_split.addWidget(cm_plot_w)

        # Right: per-class metrics table
        cls_table_w = QWidget()
        cls_table_v = QVBoxLayout(cls_table_w)
        cls_table_v.setContentsMargins(4, 0, 0, 0)
        cls_hdr = QLabel("Per-class Metrics")
        cls_hdr.setFont(QFont("Arial", 9, QFont.Bold))
        cls_table_v.addWidget(cls_hdr)
        self.tbl_cls_metrics = QTableWidget(0, 5)
        self.tbl_cls_metrics.setHorizontalHeaderLabels(
            ["Class", "Support", "Precision", "Recall", "F1"]
        )
        self.tbl_cls_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_cls_metrics.verticalHeader().setVisible(False)
        self.tbl_cls_metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cls_table_v.addWidget(self.tbl_cls_metrics)

        # Overall metrics label
        self.lbl_overall_metrics = QLabel("")
        self.lbl_overall_metrics.setStyleSheet("font-size:11px; color:#333; margin-top:4px;")
        self.lbl_overall_metrics.setWordWrap(True)
        cls_table_v.addWidget(self.lbl_overall_metrics)
        cls_table_v.addStretch()
        cm_split.addWidget(cls_table_w)
        cm_split.setSizes([350, 250])

        self.chart_tabs.addTab(cm_w, "Confusion Matrix")

        splitter.setSizes([350, 350])

        # Initial placeholder
        self._draw_placeholder()

    # ------------------------------------------------------------------ public
    def set_output_dir(self, path: str):
        """Tell the panel where to auto-save results."""
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
        path = QFileDialog.getExistingDirectory(self, "Select Results Directory", self._output_dir)
        if not path:
            return
        loaded = load_results(path)
        if not loaded:
            QMessageBox.information(self, "No Results", "No result.json files found in that directory.")
            return
        existing_ids = {r.get("run_id") for r in self._results}
        new_count = 0
        for r in loaded:
            if r.get("run_id") not in existing_ids:
                self._results.append(r)
                self._append_row(r)
                new_count += 1
        self._update_best_label()
        QMessageBox.information(self, "Loaded", f"Loaded {new_count} new run(s) from:\n{path}")

    # ------------------------------------------------------------------ table
    def _append_row(self, r: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        params = r.get("params", {})
        tm = r.get("final_test_metrics", {})
        vm = r.get("final_val_metrics", {})

        def _v(key, fmt="{:.4f}"):
            v = tm.get(key)
            return fmt.format(v) if v is not None else "—"

        def _vv(key, fmt="{:.4f}"):
            v = vm.get(key)
            return fmt.format(v) if v is not None else "—"

        train_hist = r.get("train_history", [])
        best_train = min(train_hist) if train_hist else None

        values = [
            r.get("run_id", ""),
            r.get("mode", "regression"),
            str(params.get("batch_size", "—")),
            str(params.get("learning_rate", "—")),
            str(params.get("optimizer", "—")),
            str(params.get("weight_decay", "—")),
            str(params.get("loss", "—")),
            str(r.get("best_epoch", "—")),
            f"{best_train:.4f}" if best_train is not None else "—",
            _v("rmse"),
            _v("mae"),
            _v("r2"),
            _v("accuracy", "{:.3f}"),
            _v("f1",       "{:.3f}"),
            _vv("rmse"),
            _vv("mae"),
            _vv("r2"),
            _vv("accuracy", "{:.3f}"),
            _vv("f1",       "{:.3f}"),
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
            tm = best.get("final_test_metrics", {})
            if best.get("mode") == "classification":
                acc = tm.get("accuracy")
                metric_str = f"Test Acc={acc:.3f}" if acc is not None else ""
            else:
                rmse = tm.get("rmse")
                metric_str = f"Test RMSE={rmse:.4f}" if rmse is not None else ""
            self.lbl_best.setText(f"Best: {best.get('run_id', '')}  {metric_str}")
        else:
            self.lbl_best.setText("")

        for row in range(self.table.rowCount()):
            is_best = (best and self.table.item(row, 0) and
                       self.table.item(row, 0).text() == best.get("run_id", ""))
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and not item.background().color().isValid():
                    item.setBackground(QColor("#e8f5e9") if is_best else QColor("transparent"))

    def _find_best(self):
        valid = [r for r in self._results if "error" not in r and r.get("final_test_metrics")]
        if not valid:
            return None
        if valid[0].get("mode") == "classification":
            return max(valid, key=lambda r: r["final_test_metrics"].get("accuracy", 0.0))
        return min(valid, key=lambda r: r["final_test_metrics"].get("rmse", math.inf))

    def _clear_all(self):
        if QMessageBox.question(
            self, "Clear", "Clear all results?", QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self._results.clear()
            self.table.setRowCount(0)
            self.lbl_best.setText("")
            self._draw_placeholder()

    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "No Results", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", "results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RESULT_COLS)
            for row in range(self.table.rowCount()):
                writer.writerow(
                    [self.table.item(row, col).text() if self.table.item(row, col) else ""
                     for col in range(self.table.columnCount())]
                )
        QMessageBox.information(self, "Exported", f"Results saved to {path}")

    # ------------------------------------------------------------------ plots
    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row_idx = selected[0].row()
        if row_idx < len(self._results):
            self._plot_result(self._results[row_idx])

    def _plot_result(self, r: dict):
        self._draw_loss_curves(r)
        self._draw_metrics_bar(r)
        self._draw_confusion_matrix(r)

    # -- loss curves --
    def _draw_loss_curves(self, r: dict):
        train_h = r.get("train_history", [])
        test_h  = r.get("test_history", [])

        self.ax_loss.clear()
        epochs = list(range(1, len(train_h) + 1))
        if train_h:
            self.ax_loss.plot(epochs, train_h, label="Train", linewidth=1.5, color="#1976D2")
        test_valid = [(i + 1, v) for i, v in enumerate(test_h) if v is not None]
        if test_valid:
            ex, ey = zip(*test_valid)
            self.ax_loss.plot(list(ex), list(ey), label="Test", linewidth=1.5, color="#F57C00")
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

    # -- metrics bar --
    def _draw_metrics_bar(self, r: dict):
        self.ax_val.clear()
        vm = r.get("final_val_metrics", {})
        tm = r.get("final_test_metrics", {})
        if r.get("mode") == "classification":
            metrics   = ["Accuracy", "F1"]
            test_vals = [tm.get("accuracy", 0), tm.get("f1", 0)]
            val_vals  = [vm.get("accuracy", 0), vm.get("f1", 0)]
        else:
            metrics   = ["RMSE", "MAE", "R²"]
            test_vals = [tm.get("rmse", 0), tm.get("mae", 0), tm.get("r2", 0)]
            val_vals  = [vm.get("rmse", 0), vm.get("mae", 0), vm.get("r2", 0)]
        x = range(len(metrics))
        width = 0.35
        self.ax_val.bar([xi - width/2 for xi in x], test_vals, width,
                        label="Test", color="#1976D2", alpha=0.8)
        self.ax_val.bar([xi + width/2 for xi in x], val_vals, width,
                        label="Val",  color="#388E3C", alpha=0.8)
        self.ax_val.set_xticks(list(x))
        self.ax_val.set_xticklabels(metrics)
        self.ax_val.set_title("Final Metrics")
        self.ax_val.legend(fontsize=8)
        self.ax_val.grid(True, alpha=0.3, axis="y")
        self.canvas_val.draw()

    # -- confusion matrix + per-class table --
    def _draw_confusion_matrix(self, r: dict):
        self.ax_cm.clear()
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")

        if r.get("mode") != "classification":
            self.ax_cm.text(0.5, 0.5, "Regression run\n(no confusion matrix)",
                            ha="center", va="center", transform=self.ax_cm.transAxes,
                            color="#aaa", fontsize=11)
            self.canvas_cm.draw()
            return

        # Prefer validation set; fall back to test
        vm = r.get("final_val_metrics", {})
        tm = r.get("final_test_metrics", {})
        preds  = vm.get("predictions") or tm.get("predictions")
        labels = vm.get("true_labels")  or tm.get("true_labels")
        source = "Validation" if vm.get("predictions") else "Test"
        class_names = r.get("class_names", [])

        if not preds or not labels:
            self.ax_cm.text(0.5, 0.5, "No prediction data stored\n(re-train to generate)",
                            ha="center", va="center", transform=self.ax_cm.transAxes,
                            color="#aaa", fontsize=11)
            self.canvas_cm.draw()
            return

        p = np.array(preds,  dtype=int)
        y = np.array(labels, dtype=int)
        n = len(class_names) if class_names else (max(max(p), max(y)) + 1)
        names = class_names if class_names else [str(i) for i in range(n)]

        # Build confusion matrix (row=true, col=pred)
        cm = np.zeros((n, n), dtype=int)
        for pred_i, true_i in zip(p, y):
            if 0 <= true_i < n and 0 <= pred_i < n:
                cm[true_i, pred_i] += 1

        # Plot
        im = self.ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
        self.ax_cm.set_xticks(range(n))
        self.ax_cm.set_yticks(range(n))
        self.ax_cm.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        self.ax_cm.set_yticklabels(names, fontsize=9)
        self.ax_cm.set_xlabel("Predicted", fontsize=9)
        self.ax_cm.set_ylabel("True", fontsize=9)
        self.ax_cm.set_title(f"Confusion Matrix ({source}) — {r.get('run_id', '')}",
                             fontsize=9, fontweight="bold")
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                self.ax_cm.text(
                    j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="white" if cm[i, j] > vmax * 0.6 else "black",
                )
        self.fig_cm.colorbar(im, ax=self.ax_cm, fraction=0.04, pad=0.04)
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

        # Per-class metrics
        self._fill_cls_metrics_table(cm, names, len(p))

    def _fill_cls_metrics_table(self, cm: np.ndarray, names: list, total: int):
        n = len(names)
        self.tbl_cls_metrics.setRowCount(n)
        macro_f1_sum = 0.0
        weighted_f1_sum = 0.0

        for i, name in enumerate(names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = int(cm[i, :].sum())
            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)
            macro_f1_sum += f1
            weighted_f1_sum += f1 * support

            row_vals = [name, str(support),
                        f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
            for col, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                # Colour-code F1
                if col == 4:
                    fv = float(val)
                    if fv >= 0.7:
                        item.setBackground(QColor("#c8e6c9"))
                    elif fv >= 0.4:
                        item.setBackground(QColor("#fff9c4"))
                    else:
                        item.setBackground(QColor("#ffcdd2"))
                self.tbl_cls_metrics.setItem(i, col, item)

        correct = int(np.diag(cm).sum())
        accuracy = correct / max(total, 1)
        macro_f1 = macro_f1_sum / n
        weighted_f1 = weighted_f1_sum / max(total, 1)
        self.lbl_overall_metrics.setText(
            f"Overall  accuracy={accuracy:.3f}  |  "
            f"macro-F1={macro_f1:.3f}  |  "
            f"weighted-F1={weighted_f1:.3f}  |  "
            f"n={total}"
        )

    # ------------------------------------------------------------------ placeholder
    def _draw_placeholder(self):
        for ax, canvas, title in (
            (self.ax_loss, self.canvas_loss, "Loss Curves"),
            (self.ax_val,  self.canvas_val,  "Final Metrics"),
            (self.ax_cm,   self.canvas_cm,   "Confusion Matrix"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.text(0.5, 0.5, "Select a run", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=11)
            canvas.draw()
        self.tbl_cls_metrics.setRowCount(0)
        self.lbl_overall_metrics.setText("")
