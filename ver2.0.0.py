"""
2023-12-18
ver1.6.0

1.去除信号时域和频域的直流分量
2.修复单位转换时的数值bug
"""

import sys
from PyQt5.QtWidgets import QApplication
from classes import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
