"""
2024-11-12
ver2.0.2

1.优化计算信噪比使用体验
2.修改文件区表格初始行数
3.设置切换通道步长对话框宽度
4.修改了更改数据查看时间中的起止时间逻辑
5.修复了计算滤波器阶数的 bug
6.暂时关闭了小波和小波包的重构功能
"""
import sys

from PyQt5.QtWidgets import QApplication
from utils.mainwindow import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
