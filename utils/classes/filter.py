# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/12 上午9:29
@Author  : zxy
@File    : filter.py
"""
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from scipy.signal import iircomb, iirnotch, iirpeak, cheby2, bessel, ellip, cheby1, butter, buttord, cheb1ord, cheb2ord, \
    ellipord, filtfilt

from ..widget import LineEditWithReg, Label, PushButton, ComboBox, Dialog, CheckBox


class FilterI:
    """一类滤波器"""
    filter_names = {'Butterworth': butter,
                    'Chebyshev type I': cheby1,
                    'Chebyshev type II': cheby2,
                    'Elliptic (Cauer)': ellip,
                    'Bessel/Thomson': bessel}
    btype = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    cal_names = {butter: buttord, cheby1: cheb1ord, cheby2: cheb2ord, ellip: ellipord}
    norms = ['phase', 'delay', 'mag']

    def __init__(self, filter_name: str, data: np.array):
        """
        初始化各种参数
        Args:
            filter_name: 滤波器名
            data: 数据
            
        """
        self.data = data
        self.filter_name = filter_name
        self.filter = self.filter_names[filter_name]
        self.order = 4
        self.Wn = 0.02
        self.cal_order = 0
        self.cal_Wn = 0.
        self.method = 'lowpass'
        self.wp = 0.2
        self.ws = 0.3
        self.rp = 5
        self.rs = 40
        self.gpass = 3
        self.gstop = 40
        self.analog = False
        self.norm = 'phase'
        self.flag = True

        self.dialog = None

        self.b = None
        self.a = None

        self.draw = False

        self.initDialog()

    def initDialog(self):
        """
        对话框初始化布局
        Returns:

        """
        self.dialog = Dialog()
        self.dialog.setWindowTitle(f'设计{self.filter_name}滤波器')

        combx_label = Label('滤波器类型')
        self.dialog.combx = ComboBox()
        self.dialog.combx.setToolTip('滤波器的类型')
        self.dialog.combx.addItems(self.btype)
        self.dialog.combx.currentIndexChanged.connect(self.resetCalculateParmas)

        self.dialog.checkbx = CheckBox('模拟滤波器')
        self.dialog.checkbx.setToolTip('勾选时返回模拟滤波器，否则返回数字滤波器')
        self.dialog.checkbx.stateChanged.connect(self.resetCalculateParmas)

        # 计算区域组件
        cal_label = Label('计算滤波器阶数和自然频率')
        cal_label.setAlignment(Qt.AlignHCenter)

        wp_label = Label('通带频率（wp）')
        self.dialog.wp_le = LineEditWithReg(digit=True, space=True)
        self.dialog.wp_le.setToolTip('与奈奎斯特频率的比值，选择带通或带阻滤波器时应以空格分隔截止频率比值\n'
                                     '例如：\n'
                                     'lowpass: wp = 0.2, ws = 0.3\n'
                                     'highpass: wp = 0.3, ws = 0.2\n'
                                     'bandpass: wp = 0.2 0.5, ws = 0.1 0.6\n'
                                     'bandstop: wp = 0.1 0.6, ws = 0.2 0.5')
        self.dialog.wp_le.textChanged.connect(self.resetCalculateParmas)

        ws_label = Label('阻带频率（ws）')
        self.dialog.ws_le = LineEditWithReg(digit=True, space=True)
        self.dialog.ws_le.setToolTip('与奈奎斯特频率的比值，选择带通或带阻滤波器时应以空格分隔截止频率比值\n'
                                     '例如：\n'
                                     'lowpass: wp = 0.2, ws = 0.3\n'
                                     'highpass: wp = 0.3, ws = 0.2\n'
                                     'bandpass: wp = 0.2 0.5, ws = 0.1 0.6\n'
                                     'bandstop: wp = 0.1 0.6, ws = 0.2 0.5')
        self.dialog.ws_le.textChanged.connect(self.resetCalculateParmas)

        gpass_label = Label('通带损失（gpass）')
        self.dialog.gpass_le = LineEditWithReg()
        self.dialog.gpass_le.setToolTip('通带的最大损失（dB）')
        self.dialog.gpass_le.textChanged.connect(self.resetCalculateParmas)

        gstop_label = Label('阻带衰减（gstop）')
        self.dialog.gstop_le = LineEditWithReg()
        self.dialog.gstop_le.setToolTip('阻带最小衰减（dB）')
        self.dialog.gstop_le.textChanged.connect(self.resetCalculateParmas)

        cal_order_label = Label('阶数（N）')
        self.dialog.cal_order_le = LineEditWithReg(focus=False)

        cal_Wn_label = Label('自然频率（Wn）')
        self.dialog.cal_Wn_le = LineEditWithReg(focus=False)

        self.dialog.cal_btn = PushButton('计算')
        self.dialog.cal_btn.clicked.connect(self.calculateParams)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        cal_vbox = QVBoxLayout()

        hbox1.addWidget(combx_label)
        hbox1.addWidget(self.dialog.combx)
        hbox1.addStretch(1)
        hbox1.addWidget(self.dialog.checkbx)

        hbox2.addWidget(wp_label)
        hbox2.addWidget(self.dialog.wp_le)
        hbox2.addStretch(1)
        hbox2.addWidget(ws_label)
        hbox2.addWidget(self.dialog.ws_le)

        hbox3.addWidget(gpass_label)
        hbox3.addWidget(self.dialog.gpass_le)
        hbox3.addStretch(1)
        hbox3.addWidget(gstop_label)
        hbox3.addWidget(self.dialog.gstop_le)

        hbox4.addWidget(cal_order_label)
        hbox4.addWidget(self.dialog.cal_order_le)
        hbox4.addStretch(1)
        hbox4.addWidget(cal_Wn_label)
        hbox4.addWidget(self.dialog.cal_Wn_le)

        cal_vbox.addWidget(cal_label)
        cal_vbox.addSpacing(10)
        cal_vbox.addLayout(hbox1)
        cal_vbox.addSpacing(10)
        cal_vbox.addLayout(hbox2)
        cal_vbox.addSpacing(10)
        cal_vbox.addLayout(hbox3)
        cal_vbox.addSpacing(10)
        cal_vbox.addLayout(hbox4)
        cal_vbox.addSpacing(10)
        cal_vbox.addWidget(self.dialog.cal_btn)

        # 滤波器系数组件
        filter_params_label = Label('滤波器系数')
        filter_params_label.setAlignment(Qt.AlignHCenter)

        rp_label = Label('最大纹波（rp）')
        self.dialog.rp_le = LineEditWithReg(digit=True)
        self.dialog.rp_le.setToolTip('通带最大纹波（dB）')

        rs_label = Label('最小衰减（rs）')
        self.dialog.rs_le = LineEditWithReg(digit=True)
        self.dialog.rs_le.setToolTip('阻带所需最小衰减（dB）')

        order_label = Label('阶数（N）')
        self.dialog.order_le = LineEditWithReg()
        self.dialog.order_le.setToolTip('滤波器的阶数')

        Wn_label = Label('自然频率（Wn）')
        self.dialog.Wn_le = LineEditWithReg(digit=True, space=True)
        self.dialog.Wn_le.setToolTip('与奈奎斯特频率的比值')

        btn = PushButton('确定')
        btn.clicked.connect(self.updateParams)
        btn.clicked.connect(self.design)

        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox5.addWidget(rp_label)
        hbox5.addWidget(self.dialog.rp_le)
        hbox5.addStretch(1)
        hbox5.addWidget(rs_label)
        hbox5.addWidget(self.dialog.rs_le)

        hbox6.addWidget(order_label)
        hbox6.addWidget(self.dialog.order_le)
        hbox6.addStretch(1)
        hbox6.addWidget(Wn_label)
        hbox6.addWidget(self.dialog.Wn_le)

        vbox.addSpacing(10)
        vbox.addWidget(filter_params_label)
        vbox.addSpacing(10)
        vbox.addLayout(hbox5)
        vbox.addSpacing(10)
        vbox.addLayout(hbox6)

        norm_label = Label('标准化')
        self.dialog.norm_combx = ComboBox()
        self.dialog.norm_combx.setToolTip('截止频率标准化')
        self.dialog.norm_combx.addItems(self.norms)

        if self.filter == butter:
            rp_label.setEnabled(False)
            self.dialog.rp_le.setEnabled(False)
            rs_label.setEnabled(False)
            self.dialog.rs_le.setEnabled(False)
        elif self.filter == cheby1:
            rs_label.setEnabled(False)
            self.dialog.rs_le.setEnabled(False)
        elif self.filter == cheby2:
            rp_label.setEnabled(False)
            self.dialog.rp_le.setEnabled(False)

        if self.filter == bessel:
            vbox = QVBoxLayout()
            hbox1 = QHBoxLayout()
            hbox2 = QHBoxLayout()

            hbox1.addWidget(combx_label)
            hbox1.addWidget(self.dialog.combx)
            hbox1.addStretch(1)
            hbox1.addWidget(norm_label)
            hbox1.addWidget(self.dialog.norm_combx)
            hbox1.addStretch(1)
            hbox1.addWidget(self.dialog.checkbx)
            hbox2.addWidget(order_label)
            hbox2.addWidget(self.dialog.order_le)
            hbox2.addStretch(1)
            hbox2.addWidget(Wn_label)
            hbox2.addWidget(self.dialog.Wn_le)

            vbox.addSpacing(5)
            vbox.addWidget(filter_params_label)
            vbox.addLayout(hbox1)
            vbox.addSpacing(5)
            vbox.addLayout(hbox2)
            vbox.addSpacing(5)
            vbox.addWidget(btn)
            self.dialog.setLayout(vbox)
        else:
            dialog_layout = QVBoxLayout()
            dialog_layout.addLayout(cal_vbox)
            dialog_layout.addSpacing(10)
            dialog_layout.addLayout(vbox)
            dialog_layout.addSpacing(10)
            dialog_layout.addWidget(btn)
            self.dialog.setLayout(dialog_layout)

        btn.clicked.connect(self.filterData)
        btn.clicked.connect(self.dialog.close)

    def updateParams(self):
        """
        更新滤波器参数
        Returns:

        """
        self.method = self.dialog.combx.currentText()
        self.wp = self.dialog.wp_le.text()
        self.ws = self.dialog.ws_le.text()
        self.rp = int(self.dialog.rp_le.text())
        self.rs = int(self.dialog.rs_le.text())
        self.gpass = int(self.dialog.gpass_le.text())
        self.gstop = int(self.dialog.gstop_le.text())
        self.analog = self.dialog.checkbx.isChecked()
        self.norm = self.dialog.norm_combx.currentText()
        self.order = int(self.dialog.order_le.text())
        self.Wn = self.dialog.Wn_le.text()
        self.Wn = list(map(float, self.Wn.split(' ')))
        if self.method == 'lowpass' or self.method == 'highpass':
            self.Wn = self.Wn[0]

    def calculateParams(self):
        """
        使用函数计算滤波器的阶数和自然频率并填入滤波器参数中
        Returns:

        """
        self.wp = self.dialog.wp_le.text()
        self.ws = self.dialog.ws_le.text()
        self.analog = self.dialog.checkbx.isChecked()
        self.method = self.dialog.combx.currentText()

        if self.flag:
            wp = list(map(float, self.wp.split(' ')))
            ws = list(map(float, self.ws.split(' ')))
            if self.method == 'lowpass' or self.method == 'highpass':
                wp = wp[0]
                ws = ws[0]

            self.cal_order, self.cal_Wn = self.cal_names[self.filter](wp=wp, ws=ws, gpass=self.gpass,
                                                                      gstop=self.gstop, analog=self.analog)
            if self.method == 'lowpass' or self.method == 'highpass':
                self.cal_Wn = [self.cal_Wn]

            self.dialog.cal_order_le.setText(str(self.cal_order))
            self.dialog.cal_Wn_le.setText(' '.join(map(str, self.cal_Wn)))
            self.dialog.cal_btn.setText('输入阶数与自然频率')
            self.flag = False

        else:
            self.dialog.order_le.setText(str(self.cal_order))
            self.dialog.Wn_le.setText(' '.join(map(str, self.cal_Wn)))

    def resetCalculateParmas(self):
        """
        每次参数改动后重置计算按钮
        Returns:

        """
        self.flag = True
        self.dialog.cal_order_le.setText('')
        self.dialog.cal_Wn_le.setText('')
        self.dialog.cal_btn.setText('计算')

    def design(self):
        """
        设计滤波器
        Returns:

        """
        if self.filter == butter:
            self.b, self.a = self.filter(N=self.order, Wn=self.Wn, btype=self.method, analog=self.analog)
        elif self.filter == cheby1:
            self.b, self.a = self.filter(N=self.order, rp=self.rp, Wn=self.Wn, btype=self.method, analog=self.analog)
        elif self.filter == cheby2:
            self.b, self.a = self.filter(N=self.order, rs=self.rs, Wn=self.Wn, btype=self.method, analog=self.analog)
        elif self.filter == ellip:
            self.b, self.a = self.filter(N=self.order, rp=self.rp, rs=self.rs, Wn=self.Wn, btype=self.method,
                                         analog=self.analog)
        elif self.filter == bessel:
            self.b, self.a = self.filter(N=self.order, Wn=self.Wn, btype=self.method, analog=self.analog,
                                         norm=self.norm)

    def filterData(self):
        """
        滤波数据
        Returns:

        """
        self.draw = True
        self.data = filtfilt(self.b, self.a, self.data)  # 滤波

    def runDialog(self):
        """
        更新对话框组件状态，运行对话框
        Returns:

        """
        self.draw = False
        self.dialog.combx.setCurrentText(self.method)
        self.dialog.checkbx.setChecked(self.analog)
        self.dialog.wp_le.setText(str(self.wp))
        self.dialog.ws_le.setText(str(self.ws))
        self.dialog.gpass_le.setText(str(self.gpass))
        self.dialog.gstop_le.setText(str(self.gstop))
        self.dialog.rp_le.setText(str(self.rp))
        self.dialog.rs_le.setText(str(self.rs))
        self.dialog.order_le.setText(str(self.order))
        self.dialog.Wn_le.setText(str(self.Wn))
        self.dialog.norm_combx.setCurrentText(self.norm)
        self.dialog.exec_()


class FilterII:
    """二类滤波器"""
    filter_names = {'Notch Digital Filter': iirnotch,
                    'Peak (Resonant) Digital Filter': iirpeak,
                    'Notching or Peaking Digital Comb Filter': iircomb}

    def __init__(self, filter_name: str, data: np.array):
        """
        初始化各种参数
        Args:
            filter_name: 滤波器名称
            data: 数据
            
        """
        self.data = data
        self.filter_name = filter_name
        self.filter = self.filter_names[filter_name]
        self.w0 = 0.5
        self.Q = 30
        self.fs = 2.0
        self.ftype = 'notch'
        self.pass_zero = False

        self.dialog = None

        self.b = None
        self.a = None

        self.draw = False

        self.initDialog()

    def filterData(self):
        """
        滤波数据
        Returns:

        """
        self.draw = True
        self.data = filtfilt(self.b, self.a, self.data)  # 滤波

    def initDialog(self):
        """
        初始化对话框布局
        Returns:

        """
        self.dialog = Dialog()
        self.dialog.setWindowTitle(f'设计{self.filter_name}滤波器')

        filter_params_label = Label('滤波器参数')
        filter_params_label.setAlignment(Qt.AlignHCenter)

        W0_label = Label('移除频率')
        self.dialog.W0_le = LineEditWithReg(digit=True)
        self.dialog.W0_le.setToolTip('从信号中移除的频率，值等于与奈奎斯特频率的比值')

        Q_label = Label('品质因数')
        self.dialog.Q_le = LineEditWithReg(digit=True)
        self.dialog.Q_le.setToolTip('表征notch类滤波器-3dB带宽的无量纲系数，与中心频率有关')

        btn = PushButton('确定')
        btn.clicked.connect(self.updateParams)
        btn.clicked.connect(self.design)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(W0_label)
        hbox.addWidget(self.dialog.W0_le)
        hbox.addStretch(1)
        hbox.addWidget(Q_label)
        hbox.addWidget(self.dialog.Q_le)

        vbox.addSpacing(5)
        vbox.addWidget(filter_params_label)
        vbox.addSpacing(10)
        vbox.addLayout(hbox)
        vbox.addSpacing(5)

        if self.filter == iircomb:
            ftype_label = Label('梳式滤波器类型')
            self.dialog.ftype_combx = ComboBox()
            self.dialog.ftype_combx.setToolTip(
                '生成的梳式滤波器类型，如果为notch，品质因子应用于波谷，如果为peak，品质因子应用与波峰')
            self.dialog.ftype_combx.addItems(['notch', 'peak'])

            self.dialog.pass_zero_checkbx = CheckBox('非零')
            self.dialog.pass_zero_checkbx.setToolTip('默认为否，滤波器波谷集中于频率[0, w0, 2*w0, …]，'
                                                     '波峰集中于中点[w0/2, 3*w0/2, 5*w0/2, …]；\n'
                                                     '如果为是，波谷集中于[w0/2, 3*w0/2, 5*w0/2, …]，'
                                                     '波峰集中于[0, w0, 2*w0, …]')
            hbox1 = QHBoxLayout()
            hbox1.addWidget(ftype_label)
            hbox1.addWidget(self.dialog.ftype_combx)
            hbox1.addStretch(1)
            hbox1.addWidget(self.dialog.pass_zero_checkbx)
            vbox.addLayout(hbox1)
            vbox.addSpacing(10)

        vbox.addWidget(btn)
        self.dialog.setLayout(vbox)
        btn.clicked.connect(self.filterData)
        btn.clicked.connect(self.dialog.close)

    def updateParams(self):
        """
        更新滤波器参数
        Returns:

        """
        self.draw = False
        self.w0 = float(self.dialog.W0_le.text())
        self.Q = float(self.dialog.Q_le.text())
        if self.filter == iircomb:
            self.ftype = self.dialog.ftype_combx.currentText()
            self.pass_zero = self.dialog.pass_zero_checkbx.isChecked()

    def design(self):
        """
        设计comb类滤波器
        Returns:

        """
        if self.filter == iircomb:
            self.b, self.a = self.filter(w0=self.w0, Q=self.Q, ftype=self.ftype, fs=self.fs,
                                         pass_zero=self.pass_zero)
        else:
            self.b, self.a = self.filter(w0=self.w0, Q=self.Q, fs=self.fs)

    def runDialog(self):
        """
        更新对话框组件状态，运行对话框
        Returns:

        """
        self.dialog.W0_le.setText(str(self.w0))
        self.dialog.Q_le.setText(str(self.Q))
        if self.filter == iircomb:
            self.dialog.ftype_combx.setCurrentIndex(0)
            self.dialog.pass_zero_checkbx.setChecked(self.pass_zero)
        self.dialog.exec_()


class FilterHandler:
    def __init__(self):
        """根据滤波器名创建滤波器类"""
        self.method = None
        self.filter_name = None
        self.obj = None

    def run(self, name: str, data: np.array) -> Optional[np.array]:
        """
        运行滤波器
        Returns:

        """
        if name != self.filter_name:
            self.filter_name = name
            if self.filter_name in {'Butterworth', 'Chebyshev type I', 'Chebyshev type II',
                                    'Elliptic (Cauer)', 'Bessel/Thomson'}:
                self.obj = FilterI(self.filter_name, data)
                self.method = self.obj.method
            else:
                self.obj = FilterII(self.filter_name, data)
                self.method = None
        else:
            self.obj.data = data
        self.obj.runDialog()

        return self.obj.data if self.obj.draw else None
