# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/12 上午9:28
@Author  : zxy
@File    : widget.py
"""
from cmath import inf

import numpy as np
import pyqtgraph as pg
from typing import Callable, Union, Tuple

from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator, QFont, QColor
from PyQt5.QtWidgets import QLineEdit, QLabel, QComboBox, QCheckBox, QPushButton, QRadioButton, QSpinBox, QMenu, \
    QAction, QWidget, QTextEdit, QDialog


class Menu(QMenu):
    def __init__(self,
                 parent: QWidget,
                 title: str,
                 status_tip: str = None,
                 enabled: bool = True):
        super().__init__(parent=parent, title=title)
        self.setStatusTip(status_tip)
        self.setEnabled(enabled)
        parent.addMenu(self)


class Action(QAction):
    def __init__(self,
                 parent: QWidget,
                 text: str,
                 status_tip: str,
                 slot_func: Callable,
                 shortcut: Union[int, str] = 0,
                 enabled: bool = True):
        super().__init__(parent=parent, text=text)
        self.setStatusTip(status_tip)
        self.setShortcut(shortcut)
        self.triggered.connect(slot_func)
        self.setEnabled(enabled)
        parent.addAction(self)


class Dialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)


class Label(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class PushButton(QPushButton):
    def __init__(self, text: str = None):
        super().__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class LineEdit(QLineEdit):
    def __init__(self, focus: bool = True):
        super().__init__()
        if not focus:
            self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class LineEditWithReg(LineEdit):
    """只能输入数字"""

    def __init__(self, digit: bool = False, space: bool = False, *args, **kwargs):
        """
        带正则的行输入
        Args:
            digit: 是否可输入小数点
            space: 是否可输入空格
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        if digit and space:
            reg = '[0-9. ]+'
        elif digit:
            reg = '[0-9.]+'
        elif space:
            reg = '[0-9 ]+'
        else:
            reg = '[0-9]+'
        self.regex = QRegExp(reg)
        self.validator = QRegExpValidator(self.regex)
        self.setValidator(self.validator)


class TextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class ComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('font-size: 16px; font-family: "Times New Roman", "Microsoft YaHei";')


class RadioButton(QRadioButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class CheckBox(QCheckBox):
    def __init__(self, text: str):
        super().__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class SpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


# class FigureCanvas(FigureCanvasQTAgg):
#     """设置matplotlib图可以鼠标拖动和缩放"""
#     def __init__(self, figure):
#         super(FigureCanvas, self).__init__(figure)
#         self.mouseX = 0  # 获取鼠标按下时的坐标X
#         self.mouseY = 0  # 获取鼠标按下时的坐标Y
#         self.buttonPressed = False
#
#         self.figure.canvas.mpl_connect("button_press_event", self.buttonPressEvent)
#         self.figure.canvas.mpl_connect("button_release_event", self.buttonReleaseEvent)
#         self.figure.canvas.mpl_connect("motion_notify_event", self.motionNotifyEvent)
#         self.figure.canvas.mpl_connect('scroll_event', self.scrollEvent)
#
#     def buttonPressEvent(self, event):
#         """鼠标按键按下"""
#
#         if event.inaxes:  # 判断鼠标是否在axes内
#             if event.button == 1:  # 判断按下的是否为鼠标左键1（右键是3）
#                 self.buttonPressed = True
#                 self.mouseX = event.xdata  # 获取鼠标按下时的坐标X
#                 self.mouseY = event.ydata  # 获取鼠标按下时的坐标Y
#
#     def buttonReleaseEvent(self, event):
#         """鼠标按键释放"""
#
#         if self.buttonPressed:
#             self.buttonPressed = False  # 鼠标松开，结束移动
#
#     def motionNotifyEvent(self, event):
#         """鼠标移动"""
#
#         axes = event.inaxes
#         if axes:
#             if self.buttonPressed:  # 按下状态
#                 # 计算新的坐标原点并移动
#                 # 获取当前最新鼠标坐标与按下时坐标的差值
#                 x = event.xdata - self.mouseX
#                 y = event.ydata - self.mouseY
#                 # 获取当前原点和最大点的4个位置
#                 x_min, x_max = axes.get_xlim()
#                 y_min, y_max = axes.get_ylim()
#
#                 x_min = x_min - x
#                 x_max = x_max - x
#                 y_min = y_min - y
#                 y_max = y_max - y
#
#                 axes.set_xlim(x_min, x_max)
#                 axes.set_ylim(y_min, y_max)
#                 self.figure.canvas.draw()  # 绘图动作实时反映在图像上
#
#     def scrollEvent(self, event):
#         """鼠标滚轮事件"""
#
#         axes = event.inaxes
#         x_min, x_max = axes.get_xlim()
#         y_min, y_max = axes.get_ylim()
#         if not hasattr(self, 'xfanwei'):
#             self.x_range = (x_max - x_min) / 10
#             self.y_range = (y_max - y_min) / 10
#         if event.button == 'up':
#             axes.set(xlim=(x_min + self.x_range, x_max - self.x_range))
#             axes.set(ylim=(y_min + self.y_range, y_max - self.y_range))
#         elif event.button == 'down':
#             axes.set(xlim=(x_min - self.x_range, x_max + self.x_range))
#             axes.set(ylim=(y_min - self.y_range, y_max + self.y_range))
#         self.figure.canvas.draw_idle()  # 绘图动作实时反映在图像上


class MyPlotWidget(pg.PlotWidget):
    """带字体、可显示数据"""

    def __init__(self, title: str, xlabel: str, ylabel: str, grid: bool = False, check_mouse: bool = True):
        super().__init__()
        self.check_mouse = check_mouse
        self.setTitle(f'<font face="Microsoft YaHei" size="5">{title}</font>')
        self.setLabel('bottom', f'<font face="Microsoft YaHei" size="3">{xlabel}</font>')
        self.setLabel('left', f'<font face="Microsoft YaHei" size="3">{ylabel}</font>')
        self.getAxis('bottom').setTickFont(QFont('Times New Roman'))
        self.getAxis('left').setTickFont(QFont('Times New Roman'))
        self.getAxis('left').setWidth(50)
        if check_mouse:
            self.initPlotItem(title, xlabel, ylabel, grid)
        self.sig_mouse_moved_connected = False

    def initPlotItem(self, title: str, xlabel: str, ylabel: str, grid: bool) -> None:
        """初始化一个plotitem"""
        self.plot_item = pg.PlotItem()
        self.plot_item.setTitle(f'<font face="Microsoft YaHei" size="5">{title}</font>')
        self.plot_item.setLabel('bottom', f'<font face="Microsoft YaHei" size="3">{xlabel}</font>')
        self.plot_item.setLabel('left', f'<font face="Microsoft YaHei" size="3">{ylabel}</font>')
        self.plot_item.getAxis('bottom').setTickFont(QFont('Times New Roman'))
        self.plot_item.getAxis('left').setTickFont(QFont('Times New Roman'))
        self.plot_item.getAxis('left').setWidth(50)
        if grid:
            self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.setCentralItem(self.plot_item)

    def updateAxesRange(self):
        """更新xy轴范围"""
        self.data = np.nan_to_num(np.array(self.plot_data_item.getData()))  # 获取绘图数据，并把其中可能存在的nan值替换为0
        self.xmin, self.xmax = np.min(self.data[0]), np.max(self.data[0])
        self.ymin, self.ymax = np.min(self.data[1]), np.max(self.data[1])
        self.plot_item.setXRange(self.xmin, self.xmax)
        self.plot_item.setYRange(self.ymin, self.ymax)

    def draw(self, *args, **kwargs) -> None:
        """让plotwidget中的plotitem绘图"""
        if self.check_mouse:
            self.plot_data_item = self.plot_item.plot(*args, **kwargs)
            self.updateAxesRange()
            if not self.sig_mouse_moved_connected:
                self.plot_item.scene().sigMouseMoved.connect(self.mouseMoved)  # 绘图之后绑定槽函数，否则会导致scene快速移动
                self.sig_mouse_moved_connected = True
        else:
            self.plot(*args, **kwargs)

    def mouseMoved(self, pos: Tuple) -> None:
        """鼠标移动槽函数"""
        if hasattr(self, 'text_item'):
            self.plot_item.removeItem(self.text_item)
            self.plot_item.removeItem(self.vertical_line)
            self.plot_item.removeItem(self.horizontal_line)
            self.plot_item.removeItem(self.scatter_plot_item)

        # 数据标签
        self.text_item = pg.TextItem(color=QColor('black'), border=pg.mkPen(QColor('black')),
                                     fill=pg.mkBrush(QColor('yellow')))
        self.text_item.setFont(QFont('Times New Roman', 10))
        self.plot_item.addItem(self.text_item)

        # 十字线
        self.vertical_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('black', width=0.5, style=Qt.DashLine),
                                             movable=False)
        self.horizontal_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('black', width=0.5, style=Qt.DashLine),
                                               movable=False)
        self.plot_item.addItem(self.vertical_line, ignoreBounds=True)
        self.plot_item.addItem(self.horizontal_line, ignoreBounds=True)

        # 数据点
        self.scatter_plot_item = pg.ScatterPlotItem()
        self.plot_item.addItem(self.scatter_plot_item)

        # 设置各item位置
        vb = self.plot_item.vb
        if self.plot_item.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x, y = float(mouse_point.x()), float(mouse_point.y())
            if self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax:
                index = self.searchPointIndex(x, y)
                self.scatter_plot_item.setData(pos=[[self.data[0][index], self.data[1][index]]], size=10,
                                               pen=QColor('red'))
                self.text_item.setText(f'x: {self.data[0][index]}\ny: {self.data[1][index]}')
                self.text_item.setPos(self.data[0][index], self.data[1][index])
                self.vertical_line.setPos(self.data[0][index])
                self.horizontal_line.setPos(self.data[1][index])

    def searchPointIndex(self, xpos: float, ypos: float) -> int:
        """寻找x坐标附近的点，返回该点的索引"""
        mn = inf
        idx = -1
        for i, (x, y) in enumerate(zip(*self.data)):
            dis = ((xpos - x) ** 2 + (ypos - y) ** 2) ** 0.5
            if dis < mn:
                mn = dis
                idx = i
        return idx
