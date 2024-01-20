"""
2023-1-8
ver1.7.1

1.更改表格响应signal
2.修复打包bug
"""
import sys

from PyQt5.QtWidgets import QApplication
from mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
