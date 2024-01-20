"""
2023-1-20
ver1.7.2

1.修复长时间显示数据标签时的卡顿
2.重构plot菜单结构
"""
import sys

from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
