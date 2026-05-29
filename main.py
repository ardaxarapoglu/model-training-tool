import sys
import os
import subprocess


def _bootstrap():
    """Install missing requirements on first launch, then restart so they are importable."""
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.exists(req_path):
        return

    # Fast probe — if every key package is already present, skip straight to launch
    try:
        import torch        # noqa: F401
        import torchvision  # noqa: F401
        import qtpy         # noqa: F401
        import PIL          # noqa: F401
        import numpy        # noqa: F401
        import openpyxl     # noqa: F401
        import matplotlib   # noqa: F401
        return
    except ImportError:
        pass

    print("[Bootstrap] First launch — installing requirements (this may take a few minutes)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    except subprocess.CalledProcessError:
        print("[Bootstrap] pip install failed. Please install manually:")
        print(f"  pip install -r {req_path}")
        sys.exit(1)

    # Restart so the freshly-installed packages are importable in this session
    print("[Bootstrap] Done. Restarting...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap()

os.environ.setdefault("QT_API", "pyqt5")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt

from src.gui.main_window import MainWindow
from src.utils.crash_reporter import install as _install_crash_reporter


def main():
    _install_crash_reporter("./crashes")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Froth CNN Training Tool")
    app.setOrganizationName("FrothLab")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
