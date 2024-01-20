import re

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator, QFont, QColor
from PyQt5.QtWidgets import QLineEdit, QLabel, QComboBox, QCheckBox, QPushButton, QRadioButton, QSpinBox, QHBoxLayout, \
    QVBoxLayout, QDialog, QMenu, QAction
from scipy.signal import iircomb, iirnotch, iirpeak, cheby2, bessel, ellip, cheby1, butter, buttord, cheb1ord, cheb2ord, \
    ellipord


class Menu(QMenu):

    def __init__(self, title, parent, status_tip=None, enabled=True):
        super(Menu, self).__init__(title=title, parent=parent)
        self.setStatusTip(status_tip)
        self.setEnabled(enabled)
        parent.addMenu(self)


class Action(QAction):

    def __init__(self, text, parent, status_tip, slot_func, shortcut=0):
        super(Action, self).__init__(text=text, parent=parent)
        self.setStatusTip(status_tip)
        self.setShortcut(shortcut)
        self.triggered.connect(slot_func)
        parent.addAction(self)


class Label(QLabel):

    def __init__(self, text):
        super(Label, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class PushButton(QPushButton):

    def __init__(self, text):
        super(PushButton, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class LineEdit(QLineEdit):

    def __init__(self):
        super(QLineEdit, self).__init__()
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


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
        self.setStyleSheet('font-size: 16px; font-family: "Times New Roman", "Microsoft YaHei";')


class RadioButton(QRadioButton):

    def __init__(self, text):
        super(RadioButton, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class CheckBox(QCheckBox):

    def __init__(self, text):
        super(CheckBox, self).__init__(text)
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


class SpinBox(QSpinBox):

    def __init__(self):
        super(SpinBox, self).__init__()
        self.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')


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


class MyPlotWidget(pg.PlotWidget):
    """带字体、可显示数据"""
    def __init__(self, title, xlabel, ylabel, grid=False, check_mouse=True):
        super(MyPlotWidget, self).__init__()
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

    def initPlotItem(self, title, xlabel, ylabel, grid):
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

    def draw(self, *args, **kwargs):
        """让plotwidget中的plotitem绘图"""
        if self.check_mouse:
            self.plot_data_item = self.plot_item.plot(*args, **kwargs)
            self.updateAxesRange()
            if not self.sig_mouse_moved_connected:
                self.plot_item.scene().sigMouseMoved.connect(self.mouseMoved)  # 绘图之后绑定槽函数，否则会导致scene快速移动
                self.sig_mouse_moved_connected = True
        else:
            self.plot(*args, **kwargs)

    def mouseMoved(self, pos):
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

    def searchPointIndex(self, xpos, ypos):
        """寻找x坐标附近的点，返回该点的索引"""
        # distances = []
        # for i in range(self.data.shape[1]):
        #     distances.append(np.sqrt((xpos - self.data[0][i]) ** 2 + (ypos - self.data[1][i]) ** 2))
        # index = distances.index(min(distances))
        #
        # return index

        distances = {}
        for i in range(self.data.shape[1]):
            distances[np.sqrt((xpos - self.data[0][i]) ** 2 + (ypos - self.data[1][i]) ** 2)] = i

        return distances[sorted(distances)[0]]


class FilterI(object):
    """滤波器相关的类"""
    def __init__(self, filter_name):
        """滤波器的一些参数"""
        self.filter_names = {'Butterworth': butter, 'Chebyshev type I': cheby1, 'Chebyshev type II': cheby2,
                             'Elliptic (Cauer)': ellip, 'Bessel/Thomson': bessel}
        self.filter_name = filter_name
        self.filter = self.filter_names[filter_name]
        self.order = 4
        self.Wn = 0.02
        self.btype = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        self.cal_names = {butter: buttord, cheby1: cheb1ord, cheby2: cheb2ord, ellip: ellipord}
        self.method = self.btype[0]
        self.wp = 0.2
        self.ws = 0.3
        self.rp = 5
        self.rs = 40
        self.gpass = 3
        self.gstop = 40
        self.analog = False
        self.norms = ['phase', 'delay', 'mag']
        self.norm = 'phase'
        self.flag = True

        self.initDialog()
        self.initLayout()

    def initDialog(self):
        """初始化对话框"""
        self.dialog = QDialog()
        self.dialog.setWindowTitle(f'设计{self.filter_name}滤波器')

        self.combx_label = Label('滤波器类型')
        self.combx = ComboBox()
        self.combx.setToolTip('滤波器的类型')
        self.combx.addItems(self.btype)
        self.combx.setCurrentText(self.method)
        self.combx.currentIndexChanged.connect(self.resetCalculateParmas)

        self.checkbx = CheckBox('模拟滤波器')
        self.checkbx.setToolTip('勾选时返回模拟滤波器，否则返回数字滤波器')
        self.checkbx.setChecked(self.analog)
        self.checkbx.stateChanged.connect(self.resetCalculateParmas)

        # 计算区域组件
        cal_label = Label('计算滤波器阶数和自然频率')
        cal_label.setAlignment(Qt.AlignHCenter)

        wp_label = Label('通带频率（wp）')
        self.wp_le = NumPointLineEdit()
        self.wp_le.setText(str(self.wp))
        self.wp_le.setToolTip('与带奎斯特频率的比值\n'
                              '例如：\n'
                              'lowpass: wp = 0.2, ws = 0.3\n'
                              'highpass: wp = 0.3, ws = 0.2\n'
                              'bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]\n'
                              'bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]')
        self.wp_le.textChanged.connect(self.resetCalculateParmas)

        ws_label = Label('阻带频率（ws）')
        self.ws_le = NumPointLineEdit()
        self.ws_le.setText(str(self.ws))
        self.ws_le.setToolTip('与带奎斯特频率的比值\n'
                              '例如：\n'
                              'lowpass: wp = 0.2, ws = 0.3\n'
                              'highpass: wp = 0.3, ws = 0.2\n'
                              'bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]\n'
                              'bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]')
        self.ws_le.textChanged.connect(self.resetCalculateParmas)

        gpass_label = Label('通带损失（gpass）')
        self.gpass_le = OnlyNumLineEdit()
        self.gpass_le.setText(str(self.gpass))
        self.gpass_le.setToolTip('通带的最大损失（dB）')
        self.gpass_le.textChanged.connect(self.resetCalculateParmas)

        gstop_label = Label('阻带衰减（gstop）')
        self.gstop_le = OnlyNumLineEdit()
        self.gstop_le.setText(str(self.gstop))
        self.gstop_le.setToolTip('阻带最小衰减（dB）')
        self.gstop_le.textChanged.connect(self.resetCalculateParmas)

        cal_order_label = Label('阶数（N）')
        self.cal_order_le = OnlyNumLineEdit()
        self.cal_order_le.setFocusPolicy(Qt.NoFocus)

        cal_Wn_label = Label('自然频率（Wn）')
        self.cal_Wn_le = NumPointLineEdit()
        self.cal_Wn_le.setFocusPolicy(Qt.NoFocus)

        self.cal_btn = PushButton('计算')
        self.cal_btn.clicked.connect(self.calculateParams)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        self.cal_vbox = QVBoxLayout()

        hbox1.addWidget(self.combx_label)
        hbox1.addWidget(self.combx)
        hbox1.addStretch(1)
        hbox1.addWidget(self.checkbx)

        hbox2.addWidget(wp_label)
        hbox2.addWidget(self.wp_le)
        hbox2.addStretch(1)
        hbox2.addWidget(ws_label)
        hbox2.addWidget(self.ws_le)

        hbox3.addWidget(gpass_label)
        hbox3.addWidget(self.gpass_le)
        hbox3.addStretch(1)
        hbox3.addWidget(gstop_label)
        hbox3.addWidget(self.gstop_le)

        hbox4.addWidget(cal_order_label)
        hbox4.addWidget(self.cal_order_le)
        hbox4.addStretch(1)
        hbox4.addWidget(cal_Wn_label)
        hbox4.addWidget(self.cal_Wn_le)

        self.cal_vbox.addWidget(cal_label)
        self.cal_vbox.addSpacing(10)
        self.cal_vbox.addLayout(hbox1)
        self.cal_vbox.addSpacing(10)
        self.cal_vbox.addLayout(hbox2)
        self.cal_vbox.addSpacing(10)
        self.cal_vbox.addLayout(hbox3)
        self.cal_vbox.addSpacing(10)
        self.cal_vbox.addLayout(hbox4)
        self.cal_vbox.addSpacing(10)
        self.cal_vbox.addWidget(self.cal_btn)

        # 滤波器系数组件
        self.filter_params_label = Label('滤波器系数')
        self.filter_params_label.setAlignment(Qt.AlignHCenter)

        self.rp_label = Label('最大纹波（rp）')
        self.rp_le = NumPointLineEdit()
        self.rp_le.setText(str(self.rp))
        self.rp_le.setToolTip('通带最大纹波（dB）')

        self.rs_label = Label('最小衰减（rs）')
        self.rs_le = NumPointLineEdit()
        self.rs_le.setText(str(self.rs))
        self.rs_le.setToolTip('阻带所需最小衰减（dB）')

        self.order_label = Label('阶数（N）')
        self.order_le = OnlyNumLineEdit()
        self.order_le.setText(str(self.order))
        self.order_le.setToolTip('滤波器的阶数')

        self.Wn_label = Label('自然频率（Wn）')
        self.Wn_le = NumPointLineEdit()
        self.Wn_le.setText(str(self.Wn))
        self.Wn_le.setToolTip('与奈奎斯特频率的比值')

        self.btn = PushButton('确定')
        self.btn.clicked.connect(self.update)
        self.btn.clicked.connect(self.design)

        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        self.vbox = QVBoxLayout()

        hbox5.addWidget(self.rp_label)
        hbox5.addWidget(self.rp_le)
        hbox5.addStretch(1)
        hbox5.addWidget(self.rs_label)
        hbox5.addWidget(self.rs_le)

        hbox6.addWidget(self.order_label)
        hbox6.addWidget(self.order_le)
        hbox6.addStretch(1)
        hbox6.addWidget(self.Wn_label)
        hbox6.addWidget(self.Wn_le)

        self.vbox.addSpacing(10)
        self.vbox.addWidget(self.filter_params_label)
        self.vbox.addSpacing(10)
        self.vbox.addLayout(hbox5)
        self.vbox.addSpacing(10)
        self.vbox.addLayout(hbox6)

        self.norm_label = Label('标准化')
        self.norm_combx = ComboBox()
        self.norm_combx.setToolTip('截止频率标准化')
        self.norm_combx.addItems(self.norms)
        self.norm_combx.setCurrentText(self.norm)

    def initLayout(self):
        """根据选择的滤波器种类初始化布局"""
        if self.filter == butter:
            self.rp_label.setEnabled(False)
            self.rp_le.setEnabled(False)
            self.rs_label.setEnabled(False)
            self.rs_le.setEnabled(False)
        elif self.filter == cheby1:
            self.rs_label.setEnabled(False)
            self.rs_le.setEnabled(False)
        elif self.filter == cheby2:
            self.rp_label.setEnabled(False)
            self.rp_le.setEnabled(False)
        elif self.filter == bessel:

            self.vbox = QVBoxLayout()
            hbox1 = QHBoxLayout()
            hbox2 = QHBoxLayout()

            hbox1.addWidget(self.combx_label)
            hbox1.addWidget(self.combx)
            hbox1.addStretch(1)
            hbox1.addWidget(self.norm_label)
            hbox1.addWidget(self.norm_combx)
            hbox1.addStretch(1)
            hbox1.addWidget(self.checkbx)
            hbox2.addWidget(self.order_label)
            hbox2.addWidget(self.order_le)
            hbox2.addStretch(1)
            hbox2.addWidget(self.Wn_label)
            hbox2.addWidget(self.Wn_le)

            self.vbox.addSpacing(5)
            self.vbox.addWidget(self.filter_params_label)
            self.vbox.addLayout(hbox1)
            self.vbox.addSpacing(5)
            self.vbox.addLayout(hbox2)
            self.vbox.addSpacing(5)
            self.vbox.addWidget(self.btn)
            self.dialog.setLayout(self.vbox)

    def update(self):
        """更新滤波器参数"""
        self.method = self.combx.currentText()

        self.wp = self.wp_le.text()

        self.ws = self.ws_le.text()

        self.rp = int(self.rp_le.text())

        self.rs = int(self.rs_le.text())

        self.gpass = int(self.gpass_le.text())

        self.gstop = int(self.gstop_le.text())

        self.analog = self.checkbx.isChecked()

        self.norm = self.norm_combx.currentText()

        self.order = int(self.order_le.text())

        self.Wn = self.Wn_le.text()
        if self.method == 'lowpass' or self.method == 'highpass':
            self.Wn = float(self.Wn)
        else:
            self.Wn = [float(i) for i in re.findall('[0]{1}.{1}\d+', self.Wn)]

    def calculateParams(self):
        """使用函数计算滤波器的阶数和自然频率并填入滤波器参数中"""
        self.wp = self.wp_le.text()
        self.ws = self.ws_le.text()
        self.analog = self.checkbx.isChecked()
        self.method = self.combx.currentText()

        if self.flag:
            if self.method == 'lowpass' or self.method == 'highpass':
                wp = float(self.wp)
                ws = float(self.ws)
            else:
                wp = [float(i) for i in re.findall('[0]{1}.{1}\d+', self.wp)]
                ws = [float(i) for i in re.findall('[0]{1}.{1}\d+', self.ws)]

            self.cal_order, self.cal_Wn = self.cal_names[self.filter](wp=wp, ws=ws, gpass=self.gpass,
                                                                      gstop=self.gstop, analog=self.analog)

            self.cal_order_le.setText(str(self.cal_order))
            self.cal_Wn_le.setText(str(self.cal_Wn))
            self.cal_btn.setText('输入阶数与自然频率')
            self.flag = False

        else:
            self.order_le.setText(str(self.cal_order))
            self.Wn_le.setText(str(self.cal_Wn))

    def resetCalculateParmas(self):
        """每次参数改动后重置计算按钮"""
        self.flag = True
        self.cal_order_le.setText('')
        self.cal_Wn_le.setText('')
        self.cal_btn.setText('计算')

    def design(self):
        """设计滤波器"""
        if self.filter == butter:
            self.b, self.a = self.filter(N=self.order, Wn=self.Wn, btype=self.method, analog=self.analog)
        elif self.filter == cheby1:
            self.b, self.a = self.filter(N=self.order, rp=self.rp, Wn=self.Wn, btype=self.method,
                                         analog=self.analog)
        elif self.filter == cheby2:
            self.b, self.a = self.filter(N=self.order, rs=self.rs, Wn=self.Wn, btype=self.method,
                                         analog=self.analog)
        elif self.filter == ellip:
            self.b, self.a = self.filter(N=self.order, rp=self.rp, rs=self.rs, Wn=self.Wn, btype=self.method,
                                         analog=self.analog)
        elif self.filter == bessel:
            self.b, self.a = self.filter(N=self.order, Wn=self.Wn, btype=self.method, analog=self.analog,
                                         norm=self.norm)


class FilterII(object):
    """滤波器相关的类"""

    def __init__(self, filter_name):
        """滤波器一些参数"""
        self.filter_names = {'Notch Digital Filter': iirnotch, 'Peak (Resonant) Digital Filter': iirpeak,
                             'Notching or Peaking Digital Comb Filter': iircomb}
        self.filter_name = filter_name
        self.filter = self.filter_names[filter_name]
        self.w0 = 0.5
        self.Q = 30
        self.fs = 2.0
        self.ftype = 'notch'
        self.pass_zero = False

        self.initDialog()

    def initDialog(self):
        """初始化对话框"""
        self.dialog = QDialog()
        self.dialog.setWindowTitle(f'设计{self.filter_name}滤波器')

        filter_params_label = Label('滤波器参数')
        filter_params_label.setAlignment(Qt.AlignHCenter)

        W0_label = Label('移除频率')
        self.W0_le = NumPointLineEdit()
        self.W0_le.setText(str(self.w0))
        self.W0_le.setToolTip('从信号中移除的频率，值等于与奈奎斯特频率的比值')

        Q_label = Label('品质因数')
        self.Q_le = NumPointLineEdit()
        self.Q_le.setToolTip('表征notch类滤波器-3dB带宽的无量纲系数，与中心频率有关')
        self.Q_le.setText(str(self.Q))

        self.btn = PushButton('确定')
        self.btn.clicked.connect(self.update)
        self.btn.clicked.connect(self.design)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(W0_label)
        hbox.addWidget(self.W0_le)
        hbox.addStretch(1)
        hbox.addWidget(Q_label)
        hbox.addWidget(self.Q_le)

        vbox.addSpacing(5)
        vbox.addWidget(filter_params_label)
        vbox.addSpacing(10)
        vbox.addLayout(hbox)
        vbox.addSpacing(5)

        if self.filter == iircomb:
            ftype_label = Label('梳式滤波器类型')
            self.ftype_combx = ComboBox()
            self.ftype_combx.setToolTip('生成的梳式滤波器类型，如果为notch，品质因子应用于波谷，如果为peak，品质因子应用与波峰')
            self.ftype_combx.addItems(['notch', 'peak'])
            self.ftype_combx.setCurrentIndex(0)

            self.pass_zero_checkbx = CheckBox('非零')
            self.pass_zero_checkbx.setToolTip('默认为否，滤波器波谷集中于频率[0, w0, 2*w0, …]，'
                                              '波峰集中于中点[w0/2, 3*w0/2, 5*w0/2, …]；\n'
                                              '如果为是，波谷集中于[w0/2, 3*w0/2, 5*w0/2, …]，'
                                              '波峰集中于[0, w0, 2*w0, …]')
            hbox1 = QHBoxLayout()
            hbox1.addWidget(ftype_label)
            hbox1.addWidget(self.ftype_combx)
            hbox1.addStretch(1)
            hbox1.addWidget(self.pass_zero_checkbx)
            vbox.addLayout(hbox1)
            vbox.addSpacing(10)

        vbox.addWidget(self.btn)
        self.dialog.setLayout(vbox)

    def update(self):
        """更新滤波器参数"""
        self.w0 = float(self.W0_le.text())

        self.Q = float(self.Q_le.text())

        if self.filter == iircomb:
            self.ftype = self.ftype_combx.currentText()
            self.pass_zero = self.pass_zero_checkbx.isChecked()

    def design(self):
        """设计comb类滤波器"""
        if self.filter == iircomb:
            self.b, self.a = self.filter(w0=self.w0, Q=self.Q, ftype=self.ftype, fs=self.fs,
                                         pass_zero=self.pass_zero)
        else:
            self.b, self.a = self.filter(w0=self.w0, Q=self.Q, fs=self.fs)
