import sys
import os

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
