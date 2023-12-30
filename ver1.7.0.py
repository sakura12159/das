"""
2023-12-30
ver1.7.0

1.重构代码整体结构
"""

import sys
from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
