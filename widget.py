"""重写组件"""

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator, QFont
from PyQt5.QtWidgets import QLineEdit, QLabel, QComboBox, QCheckBox, QPushButton, QRadioButton
from pyqtgraph import PlotWidget


class Label(QLabel):

    def __init__(self, text):
        super(Label, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')


class PushButton(QPushButton):

    def __init__(self, text):
        super(PushButton, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')


class LineEdit(QLineEdit):

    def __init__(self):
        super(QLineEdit, self).__init__()
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')


class OnlyNumLineEdit(LineEdit):
    """只能输入数字和"""

    def __init__(self):
        super(OnlyNumLineEdit, self).__init__()
        self.regex = QRegExp('[0-9]+')
        self.validator = QRegExpValidator(self.regex)
        self.setValidator(self.validator)


class NumPointLineEdit(LineEdit):
    """只能输入数字和."""

    def __init__(self):
        super(NumPointLineEdit, self).__init__()
        self.regex = QRegExp('[0-9. ]+')
        self.validator = QRegExpValidator(self.regex)
        self.setValidator(self.validator)


class ComboBox(QComboBox):

    def __init__(self):
        super(ComboBox, self).__init__()
        self.setStyleSheet('font-size: 16px; font-family: "Times New Roman";')


class RadioButton(QRadioButton):

    def __init__(self, text):
        super(RadioButton, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')


class CheckBox(QCheckBox):

    def __init__(self, text):
        super(CheckBox, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')


# class FigureCanvas(FigureCanvasQTAgg):
#     """设置matplotlib图可以鼠标拖动和缩放"""
#
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


class MyPlotWidget(PlotWidget):
    """带字体的pw"""

    def __init__(self, title, xlabel, ylabel, grid=False):
        super(MyPlotWidget, self).__init__()
        self.setTitle(f'<font face="Times New Roman" size="5">{title}</font>')
        self.setLabel('bottom', f'<font face="Times New Roman">{xlabel}</font>')
        self.setLabel('left', f'<font face="Times New Roman">{ylabel}</font>')
        self.getAxis('bottom').setTickFont(QFont('Times New Roman'))
        self.getAxis('left').setTickFont(QFont('Times New Roman'))
        self.getAxis('left').setWidth(50)
        if grid:
            self.showGrid(x=True, y=True, alpha=0.2)
