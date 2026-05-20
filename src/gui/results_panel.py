"""Results panel: table of all runs + loss curves + metrics comparison."""
import os
import csv
import math

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QGroupBox, QSplitter, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QScrollArea, QFrame,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


RESULT_COLS = [
    "Run ID", "Batch", "LR", "Optimizer", "WD", "Loss Fn",
    "Best Ep.", "Train Loss", "Test RMSE", "Test MAE", "Test R²",
    "Val RMSE", "Val MAE", "Val R²", "Time (s)", "Checkpoint",
]


class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._results = []
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
        self.lbl_best = QLabel("")
        self.lbl_best.setStyleSheet("font-weight:bold;color:#1b5e20;")
        toolbar.addWidget(self.btn_clear)
        toolbar.addWidget(self.btn_export)
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

        # ---- Charts ----
        charts_widget = QWidget()
        ch = QHBoxLayout(charts_widget)
        ch.setContentsMargins(0, 0, 0, 0)

        self.fig_loss = Figure(figsize=(5, 3), tight_layout=True)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        ch.addWidget(self.canvas_loss)

        self.fig_val = Figure(figsize=(4, 3), tight_layout=True)
        self.ax_val = self.fig_val.add_subplot(111)
        self.canvas_val = FigureCanvas(self.fig_val)
        ch.addWidget(self.canvas_val)

        splitter.addWidget(charts_widget)
        splitter.setSizes([350, 300])

        # Initial empty plots
        self._draw_placeholder()

    # ------------------------------------------------------------------ public
    def add_results(self, results: list):
        for r in results:
            self._results.append(r)
            self._append_row(r)
        self._update_best_label()

    def add_result(self, result: dict):
        self._results.append(result)
        self._append_row(result)
        self._update_best_label()

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
            _vv("rmse"),
            _vv("mae"),
            _vv("r2"),
            f"{r.get('elapsed_seconds', 0):.1f}",
            os.path.basename(r.get("checkpoint_path", "")),
        ]

        for col, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        # Error rows get red background
        if "error" in r:
            for col in range(len(RESULT_COLS)):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QColor("#ffcdd2"))

    def _update_best_label(self):
        best = self._find_best()
        if best:
            tm = best.get("final_test_metrics", {})
            rmse = tm.get("rmse")
            self.lbl_best.setText(
                f"Best: {best.get('run_id', '')}  "
                f"Test RMSE={rmse:.4f}" if rmse else f"Best: {best.get('run_id', '')}"
            )
        else:
            self.lbl_best.setText("")

        # Highlight best row
        for row in range(self.table.rowCount()):
            is_best = best and self.table.item(row, 0) and self.table.item(row, 0).text() == best.get("run_id", "")
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and not item.background().color().isValid():
                    if is_best:
                        item.setBackground(QColor("#e8f5e9"))
                    else:
                        item.setBackground(QColor("transparent"))

    def _find_best(self):
        valid = [r for r in self._results if "error" not in r and r.get("final_test_metrics")]
        if not valid:
            return None
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
                    [self.table.item(row, col).text() if self.table.item(row, col) else "" for col in range(self.table.columnCount())]
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
            self.ax_loss.axvline(best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best ep={best_ep}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title(f"Loss — {r.get('run_id', '')}")
        self.ax_loss.legend(fontsize=8)
        self.ax_loss.grid(True, alpha=0.3)
        self.canvas_loss.draw()

        # Validation bar chart
        self.ax_val.clear()
        vm = r.get("final_val_metrics", {})
        tm = r.get("final_test_metrics", {})
        metrics = ["RMSE", "MAE", "R²"]
        test_vals = [tm.get("rmse", 0), tm.get("mae", 0), tm.get("r2", 0)]
        val_vals  = [vm.get("rmse", 0), vm.get("mae", 0), vm.get("r2", 0)]
        x = range(len(metrics))
        width = 0.35
        self.ax_val.bar([xi - width/2 for xi in x], test_vals, width, label="Test",  color="#1976D2", alpha=0.8)
        self.ax_val.bar([xi + width/2 for xi in x], val_vals,  width, label="Val",   color="#388E3C", alpha=0.8)
        self.ax_val.set_xticks(list(x))
        self.ax_val.set_xticklabels(metrics)
        self.ax_val.set_title("Final Metrics")
        self.ax_val.legend(fontsize=8)
        self.ax_val.grid(True, alpha=0.3, axis="y")
        self.canvas_val.draw()

    def _draw_placeholder(self):
        for ax, canvas, title in (
            (self.ax_loss, self.canvas_loss, "Loss Curves"),
            (self.ax_val,  self.canvas_val,  "Final Metrics"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.text(0.5, 0.5, "Select a run", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=11)
            canvas.draw()
