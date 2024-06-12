"""
2024-6-12
ver1.8.0

1.添加了筛选数据的功能
2.重构了滤波器部分的代码结构
3.更换了部分图片
4.修复一些 bug 和拼写错误
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
