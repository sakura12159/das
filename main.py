"""
2024-8-17
ver1.8.3

1.修改了计算信噪比功能中的 bug
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
