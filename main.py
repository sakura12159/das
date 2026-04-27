"""
2024-2-19
ver2.0.3

1.修改了表格区域查看文件的功能
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

"pyinstaller -F -w -i image/favicon.ico main.py"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
