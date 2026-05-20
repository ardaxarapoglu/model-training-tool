"""Main application window."""
import os

from qtpy.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QAction, QFileDialog, QMessageBox, QLabel,
)
from qtpy.QtCore import Qt, QSettings
from qtpy.QtGui import QKeySequence, QFont

from .experiments_panel import ExperimentsPanel
from .preprocessing_panel import PreprocessingPanel
from .model_panel import ModelPanel
from .training_panel import TrainingPanel
from .results_panel import ResultsPanel
from ..utils import config_manager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Froth CNN Training Tool")
        self.setMinimumSize(1280, 820)
        self._config_path = None
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._restore_geometry()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vl = QVBoxLayout(central)
        vl.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setDocumentMode(True)
        vl.addWidget(self.tabs)

        self.exp_panel   = ExperimentsPanel()
        self.prep_panel  = PreprocessingPanel()
        self.model_panel = ModelPanel()
        self.train_panel = TrainingPanel()
        self.res_panel   = ResultsPanel()

        self.tabs.addTab(self.exp_panel,   "① Experiments")
        self.tabs.addTab(self.prep_panel,  "② Preprocessing")
        self.tabs.addTab(self.model_panel, "③ Model")
        self.tabs.addTab(self.train_panel, "④ Training")
        self.tabs.addTab(self.res_panel,   "⑤ Results")

        for i in range(5):
            self.tabs.setTabToolTip(i, [
                "Define experiments, import xlsx forms, assign train/test/validation splits",
                "Configure cropping, resizing, and augmentation",
                "Choose transfer-learning architecture or build a custom CNN",
                "Set hyperparameters, enable grid search, run training",
                "View metrics, loss curves, and compare runs",
            ][i])

        self.train_panel.run_requested.connect(self._on_run_training)

    def _setup_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("&File")

        act = QAction("&New Project", self)
        act.setShortcut(QKeySequence.New)
        act.triggered.connect(self._new_project)
        file_m.addAction(act)

        act = QAction("&Open Project…", self)
        act.setShortcut(QKeySequence.Open)
        act.triggered.connect(self._open_project)
        file_m.addAction(act)

        act = QAction("&Save Project", self)
        act.setShortcut(QKeySequence.Save)
        act.triggered.connect(self._save_project)
        file_m.addAction(act)

        act = QAction("Save Project &As…", self)
        act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act.triggered.connect(self._save_as)
        file_m.addAction(act)

        file_m.addSeparator()

        act = QAction("E&xit", self)
        act.setShortcut(QKeySequence.Quit)
        act.triggered.connect(self.close)
        file_m.addAction(act)

        tools_m = mb.addMenu("&Tools")

        act = QAction("Export &Label CSV…", self)
        act.setToolTip("Generate CSV mapping images to PB concentrations")
        act.triggered.connect(self._export_csv)
        tools_m.addAction(act)

        act = QAction("Count Images per Split", self)
        act.triggered.connect(self._count_images)
        tools_m.addAction(act)

        help_m = mb.addMenu("&Help")
        act = QAction("&About", self)
        act.triggered.connect(self._about)
        help_m.addAction(act)

    def _setup_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = QLabel("Ready")
        sb.addWidget(self.lbl_status)

    def _restore_geometry(self):
        settings = QSettings("FrothLab", "FrothCNNTool")
        geo = settings.value("mainwindow/geometry")
        if geo:
            self.restoreGeometry(geo)

    # ------------------------------------------------------------------ config
    def _full_config(self) -> dict:
        return {
            "experiments":  self.exp_panel.get_config(),
            "preprocessing": self.prep_panel.get_config(),
            "model":        self.model_panel.get_config(),
            "training":     self.train_panel.get_config(),
        }

    def _apply_config(self, cfg: dict):
        if "experiments"  in cfg: self.exp_panel.set_config(cfg["experiments"])
        if "preprocessing" in cfg: self.prep_panel.set_config(cfg["preprocessing"])
        if "model"        in cfg: self.model_panel.set_config(cfg["model"])
        if "training"     in cfg: self.train_panel.set_config(cfg["training"])

    # ------------------------------------------------------------------ menu actions
    def _new_project(self):
        if QMessageBox.question(
            self, "New Project", "Clear all current settings?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self._apply_config(config_manager.default_config())
            self._config_path = None
            self.setWindowTitle("Froth CNN Training Tool")
            self.lbl_status.setText("New project")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON (*.json)")
        if not path:
            return
        try:
            cfg = config_manager.load(path)
            self._apply_config(cfg)
            self._config_path = path
            self.setWindowTitle(f"Froth CNN Training Tool — {os.path.basename(path)}")
            self.lbl_status.setText(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open project:\n{e}")

    def _save_project(self):
        if self._config_path:
            self._do_save(self._config_path)
        else:
            self._save_as()

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "JSON (*.json)")
        if path:
            if not path.endswith(".json"):
                path += ".json"
            self._do_save(path)

    def _do_save(self, path: str):
        try:
            config_manager.save(self._full_config(), path)
            self._config_path = path
            self.setWindowTitle(f"Froth CNN Training Tool — {os.path.basename(path)}")
            self.lbl_status.setText(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save project:\n{e}")

    def _export_csv(self):
        from ..utils.csv_handler import export_labels
        experiments = self.exp_panel.get_config()
        if not experiments:
            QMessageBox.warning(self, "No Experiments", "Define experiments first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Label CSV", "labels.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            n = export_labels(experiments, path)
            self.lbl_status.setText(f"CSV exported: {n} rows → {path}")
            QMessageBox.information(self, "Exported", f"{n} image-label rows written to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"CSV export failed:\n{e}")

    def _count_images(self):
        from ..utils.csv_handler import count_images
        experiments = self.exp_panel.get_config()
        counts = count_images(experiments)
        msg = "\n".join(
            f"{split.capitalize():12s}: {count:>5} images"
            for split, count in counts.items()
        )
        QMessageBox.information(self, "Image Counts", msg)

    def _about(self):
        QMessageBox.about(
            self,
            "About Froth CNN Training Tool",
            "<b>Froth CNN Training Tool</b><br>"
            "Batch CNN training for froth flotation Pb-concentration regression.<br><br>"
            "Supports transfer learning, custom CNNs, data augmentation, "
            "and grid-search hyperparameter optimisation.<br><br>"
            "<i>Validation set is held out and never used during training or model "
            "selection — evaluated once for final reporting only.</i>",
        )

    # ------------------------------------------------------------------ training bridge
    def _on_run_training(self):
        cfg = self._full_config()
        experiments = cfg.get("experiments", [])
        val_exps = [e for e in experiments if e.get("split") == "validation"]

        # Hard validation set safety check
        if val_exps:
            QMessageBox.information(
                self,
                "Validation Set Isolated",
                f"{len(val_exps)} validation experiment(s) are held out.\n"
                "They will NOT be used during training or model selection.\n"
                "They are evaluated once at the very end of each run.",
            )

        self.train_panel.start_training(cfg)
        self.tabs.setCurrentWidget(self.train_panel)

    def receive_training_results(self, results: list):
        """Called by TrainingPanel when training finishes."""
        self.res_panel.add_results(results)
        self.tabs.setCurrentWidget(self.res_panel)
        self.lbl_status.setText(f"Training complete — {len(results)} run(s)")

    # ------------------------------------------------------------------ close
    def closeEvent(self, event):
        QSettings("FrothLab", "FrothCNNTool").setValue(
            "mainwindow/geometry", self.saveGeometry()
        )
        super().closeEvent(event)
