"""
2024-10-05
ver2.0.1

1.删除了多余文件，修改了数据采集参数的显示内容
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
