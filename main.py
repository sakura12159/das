"""
2024-6-6
ver1.7.4

1.修复更改读取文件数后灰度图显示范围的问题
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
