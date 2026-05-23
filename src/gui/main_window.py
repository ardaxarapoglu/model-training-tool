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
        self._try_load_default_project()

    # ------------------------------------------------------------------ default project
    def _try_load_default_project(self):
        """Auto-open default_project.json if it exists next to main.py (project root)."""
        # main_window.py lives in src/gui/ — walk up two levels to get the project root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(root, "default_project.json")
        if not os.path.isfile(default_path):
            return
        try:
            cfg = config_manager.load(default_path)
            self._apply_config(cfg)
            self._config_path = default_path
            self.setWindowTitle(f"Froth CNN Training Tool — default_project.json")
            self.lbl_status.setText(f"Loaded default project: {default_path}")
        except Exception as exc:
            self.lbl_status.setText(f"Could not load default_project.json: {exc}")

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
        model_cfg = self.model_panel.get_config()
        # classification lives in model_cfg["classification"]; surface it top-level for trainer
        return {
            "experiments":    self.exp_panel.get_config(),
            "preprocessing":  self.prep_panel.get_config(),
            "model":          model_cfg,
            "classification": model_cfg.get("classification", {"enabled": False, "classes": []}),
            "training":       self.train_panel.get_config(),
        }

    def _apply_config(self, cfg: dict):
        if "experiments"   in cfg: self.exp_panel.set_config(cfg["experiments"])
        if "preprocessing" in cfg: self.prep_panel.set_config(cfg["preprocessing"])
        if "model"         in cfg:
            model_cfg = cfg["model"]
            # Back-fill classification from top-level key if present (old saves)
            if "classification" in cfg and "classification" not in model_cfg:
                model_cfg = dict(model_cfg)
                model_cfg["classification"] = cfg["classification"]
            self.model_panel.set_config(model_cfg)
        if "training"      in cfg: self.train_panel.set_config(cfg["training"])

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
            model_cfg = self.model_panel.get_config()
            class_cfg = model_cfg.get("classification", {"enabled": False, "classes": []})
            n = export_labels(experiments, path, class_cfg=class_cfg)
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

        # Pre-flight validation — block if there are hard errors
        errors, warnings = self._validate_training_config(cfg)
        if errors:
            QMessageBox.critical(
                self, "Cannot Start Training",
                "Fix the following issues before training:\n\n" +
                "\n".join(f"• {e}" for e in errors),
            )
            return

        if warnings:
            reply = QMessageBox.warning(
                self, "Training Warnings",
                "Warnings (training will continue but results may be affected):\n\n" +
                "\n".join(f"• {w}" for w in warnings) +
                "\n\nProceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Test-set isolation notice (test = held out, evaluated once at the end)
        experiments = cfg.get("experiments", [])
        test_exps = [e for e in experiments if e.get("split") == "test"]
        if test_exps:
            QMessageBox.information(
                self,
                "Test Set Isolated",
                f"{len(test_exps)} test experiment(s) are held out.\n"
                "They will NOT be used during training or model selection.\n"
                "They are evaluated exactly once at the very end of each run.",
            )

        # Auto-export label CSV to output dir before starting
        out_dir = cfg["training"].get("output_dir", "./results")
        csv_path = os.path.join(out_dir, "labels.csv")
        try:
            os.makedirs(out_dir, exist_ok=True)
            from ..utils.csv_handler import export_labels
            n_rows = export_labels(cfg["experiments"], csv_path,
                                   class_cfg=cfg.get("classification", {}))
            self.lbl_status.setText(f"Labels exported: {n_rows} time frames → {csv_path}")
        except Exception as exc:
            self.lbl_status.setText(f"Warning: could not auto-export label CSV: {exc}")

        self.train_panel.start_training(cfg)
        self.tabs.setCurrentWidget(self.train_panel)

    @staticmethod
    def _validate_training_config(cfg: dict):
        """Return (errors, warnings) lists.  Errors block training; warnings prompt."""
        from ..core.dataset import collect_samples
        errors = []
        warnings = []

        experiments = cfg.get("experiments", [])
        class_cfg   = cfg.get("classification", {"enabled": False, "classes": []})

        if not experiments:
            errors.append("No experiments defined.  Go to the Experiments tab and add at least one.")
            return errors, warnings

        # Count usable images per split
        # validation = monitored during training; test = held out
        split_counts = {}
        missing_folders = []
        for split in ("train", "validation", "test"):
            try:
                samples = collect_samples(experiments, split, class_cfg)
                split_counts[split] = len(samples)
            except Exception as exc:
                errors.append(f"Error scanning {split} samples: {exc}")
                split_counts[split] = 0

        # Check for experiments with no folder paths set
        for exp in experiments:
            for tf in exp.get("time_frames", []):
                folder = tf.get("folder_path", "")
                if folder and not __import__("os").path.isdir(folder):
                    missing_folders.append(
                        f"{exp.get('id','?')} / {tf.get('name','?')}: {folder}"
                    )

        if missing_folders:
            snippet = "\n  ".join(missing_folders[:5])
            suffix  = f"\n  … and {len(missing_folders)-5} more" if len(missing_folders) > 5 else ""
            warnings.append(
                f"{len(missing_folders)} folder path(s) do not exist (skipped):\n  {snippet}{suffix}"
            )

        if split_counts.get("train", 0) == 0:
            errors.append(
                "No training images found.  Make sure at least one experiment is set to "
                "'train' and its time-frame folder paths are correct."
            )

        if split_counts.get("validation", 0) == 0:
            warnings.append(
                "No validation images found.  Early stopping will use training loss, "
                "which may cause overfitting.  Assign at least one experiment to 'Validation'."
            )

        # Classification-specific checks
        if class_cfg.get("enabled", False):
            classes = class_cfg.get("classes", [])
            if len(classes) < 2:
                errors.append("Classification is enabled but fewer than 2 classes are defined.")
            else:
                # Check boundaries are ascending
                bounds = [c.get("max") for c in classes[:-1]]
                if any(b is None for b in bounds):
                    errors.append(
                        "A non-final class has no upper bound set.  "
                        "Only the last class should have an empty upper bound."
                    )
                elif bounds != sorted(bounds):
                    errors.append("Class boundaries are not in ascending order.")

        return errors, warnings

    def receive_training_results(self, results: list):
        """Called by TrainingPanel when training finishes."""
        # Keep the results panel's output_dir in sync with current training config
        out_dir = self.train_panel.get_config().get("output_dir", "./results")
        self.res_panel.set_output_dir(out_dir)
        self.res_panel.add_results(results)
        self.tabs.setCurrentWidget(self.res_panel)
        self.lbl_status.setText(f"Training complete — {len(results)} run(s)")

    # ------------------------------------------------------------------ close
    def closeEvent(self, event):
        QSettings("FrothLab", "FrothCNNTool").setValue(
            "mainwindow/geometry", self.saveGeometry()
        )
        super().closeEvent(event)
