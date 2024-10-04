"""
2024-10-04
ver2.0.0

1.优化代码逻辑，重构代码结构
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
