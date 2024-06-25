"""
2024-6-25
ver1.0.0

1.新增了基于 Scouter 采集软件的数据读取
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow_scouter import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
