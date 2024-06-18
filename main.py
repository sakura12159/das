"""
2024-6-18
ver1.8.1

1.修改了 data_sift 部分的逻辑
2.修改了 filter 部分的逻辑
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
