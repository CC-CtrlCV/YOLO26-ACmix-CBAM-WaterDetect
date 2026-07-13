import sys

from PyQt5.QtWidgets import QApplication

from yolo_water_detect.ui.main_ui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
