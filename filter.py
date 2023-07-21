"""定义滤波器"""

import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QDialog
from scipy.signal import iircomb, iirnotch, iirpeak, cheby2, bessel, ellip, cheby1, butter, buttord, cheb1ord, cheb2ord, \
    ellipord

from function import printError
from widget import *


class FilterI(object):
    """滤波器相关的类"""

    def __init__(self, name):
        """滤波器的一些参数"""

        self.names = {'Butterworth': butter, 'Chebyshev type I': cheby1, 'Chebyshev type II': cheby2,
                      'Elliptic (Cauer)': ellip, 'Bessel/Thomson': bessel}
        self.name = name
        self.filter = self.names[name]
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
        self.dialog.setWindowTitle(f'{self.name} Filter')

        self.combx_label = Label('Filter Type')
        self.combx = ComboBox()
        self.combx.setToolTip('The type of filter')
        self.combx.addItems(self.btype)
        self.combx.setCurrentText(self.method)
        self.combx.currentIndexChanged.connect(self.resetCalculateParmas)

        self.checkbx = CheckBox('Analog Filter')
        self.checkbx.setToolTip('When True, return an analog filter, otherwise a digital filter is returned')
        self.checkbx.setChecked(self.analog)
        self.checkbx.stateChanged.connect(self.resetCalculateParmas)

        # 计算区域组件
        cal_label = Label('Calculate Order And Natural Frequency:')
        cal_label.setAlignment(Qt.AlignHCenter)

        wp_label = Label('Passband Edge Frequency(wp)')
        self.wp_le = NumPointLineEdit()
        self.wp_le.setText(str(self.wp))
        self.wp_le.setToolTip('Ratio to Nyquist Frequency\n'
                              'For example:\n'
                              'Lowpass: wp = 0.2, ws = 0.3\n'
                              'Highpass: wp = 0.3, ws = 0.2\n'
                              'Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]\n'
                              'Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]')
        self.wp_le.textChanged.connect(self.resetCalculateParmas)

        ws_label = Label('Stopband Edge Frequency(ws)')
        self.ws_le = NumPointLineEdit()
        self.ws_le.setText(str(self.ws))
        self.ws_le.setToolTip('Ratio to Nyquist Frequency\n'
                              'For example:\n'
                              'Lowpass: wp = 0.2, ws = 0.3\n'
                              'Highpass: wp = 0.3, ws = 0.2\n'
                              'Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]\n'
                              'Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]')
        self.ws_le.textChanged.connect(self.resetCalculateParmas)

        gpass_label = Label('Passband Loss(gpass)')
        self.gpass_le = OnlyNumLineEdit()
        self.gpass_le.setText(str(self.gpass))
        self.gpass_le.setToolTip('The maximum loss in the passband (dB)')
        self.gpass_le.textChanged.connect(self.resetCalculateParmas)

        gstop_label = Label('Stopband Attenuation(gstop)')
        self.gstop_le = OnlyNumLineEdit()
        self.gstop_le.setText(str(self.gstop))
        self.gstop_le.setToolTip('The minimum attenuation in the stopband (dB)')
        self.gstop_le.textChanged.connect(self.resetCalculateParmas)

        cal_order_label = Label('Order(N)')
        self.cal_order_le = OnlyNumLineEdit()
        self.cal_order_le.setFocusPolicy(Qt.NoFocus)

        cal_Wn_label = Label('Natural Frequency(Wn)')
        self.cal_Wn_le = NumPointLineEdit()
        self.cal_Wn_le.setFocusPolicy(Qt.NoFocus)

        self.cal_btn = PushButton('Calculate')
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
        self.filter_params_label = Label('Filter Parameters')
        self.filter_params_label.setAlignment(Qt.AlignHCenter)

        self.rp_label = Label('Maximum Ripple(rp)')
        self.rp_le = NumPointLineEdit()
        self.rp_le.setText(str(self.rp))
        self.rp_le.setToolTip(
            'The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number')

        self.rs_label = Label('Minimum Attenuation(rs)')
        self.rs_le = NumPointLineEdit()
        self.rs_le.setText(str(self.rs))
        self.rs_le.setToolTip(
            'The minimum attenuation required in the stop band. Specified in decibels, as a positive number')

        self.order_label = Label('Order(N)')
        self.order_le = OnlyNumLineEdit()
        self.order_le.setText(str(self.order))
        self.order_le.setToolTip('The order of the filter')

        self.Wn_label = Label('Natural Frequency(Wn)')
        self.Wn_le = NumPointLineEdit()
        self.Wn_le.setText(str(self.Wn))
        self.Wn_le.setToolTip(
            'A scalar or length-2 sequence giving the critical frequencies (defined by the norm parameter)\n'
            'For analog filters, Wn is an angular frequency (e.g., rad/s)\n'
            'For digital filters, Wn are in the same units as fs\n'
            'By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1,\n'
            'where 1 is the Nyquist frequency, (Wn is thus in half-cycles / sample.)')

        self.btn = PushButton('OK')
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

        self.norm_label = Label('Norm')
        self.norm_combx = ComboBox()
        self.norm_combx.setToolTip('Critical frequency normalization')
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

            try:
                self.cal_order, self.cal_Wn = self.cal_names[self.filter](wp=wp, ws=ws, gpass=self.gpass,
                                                                          gstop=self.gstop, analog=self.analog)
            except Exception as err:
                printError(err)

            self.cal_order_le.setText(str(self.cal_order))
            self.cal_Wn_le.setText(str(self.cal_Wn))
            self.cal_btn.setText('Input Order And Natural Frequency')
            self.flag = False

        else:
            self.order_le.setText(str(self.cal_order))
            self.Wn_le.setText(str(self.cal_Wn))

    def resetCalculateParmas(self):
        """每次参数改动后重置计算按钮"""

        self.flag = True
        self.cal_order_le.setText('')
        self.cal_Wn_le.setText('')
        self.cal_btn.setText('Calculate')

    def design(self):
        """设计滤波器"""
        try:
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
        except Exception as err:
            printError(err)


class FilterII(object):
    """滤波器相关的类"""

    def __init__(self, name):
        """滤波器一些参数"""

        self.names = {'Notch Digital Filter': iirnotch, 'Peak (Resonant) Digital Filter': iirpeak,
                      'Notching or Peaking Digital Comb Filter': iircomb}
        self.name = name
        self.filter = self.names[name]
        self.w0 = 0.5
        self.Q = 30
        self.fs = 2.0
        self.ftype = 'notch'
        self.pass_zero = False

        self.initDialog()

    def initDialog(self):
        """初始化对话框"""

        self.dialog = QDialog()
        self.dialog.setWindowTitle(f'{self.name} Filter')

        filter_params_label = Label('Filter Parameters')
        filter_params_label.setAlignment(Qt.AlignHCenter)

        W0_label = Label('Frequency to remove')
        self.W0_le = NumPointLineEdit()
        self.W0_le.setText(str(self.w0))
        self.W0_le.setToolTip('Frequency to remove from a signal\n'
                              'If fs is specified, this is in the same units as fs. By default,\n'
                              'it is a normalized scalar that must satisfy 0 < w0 < 1,\n'
                              'with w0 = 1 corresponding to half of the sampling frequency')

        Q_label = Label('Quality Factor')
        self.Q_le = NumPointLineEdit()
        self.Q_le.setToolTip('Dimensionless parameter that characterizes notch filter -3 dB bandwidth bw\n'
                             'relative to its center frequency, Q = w0/bw')
        self.Q_le.setText(str(self.Q))

        self.btn = PushButton('OK')
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
            ftype_label = Label('Type of Comb Filter')
            self.ftype_combx = ComboBox()
            self.ftype_combx.setToolTip('The type of comb filter generated by the function\n'
                                        'If ‘notch’, then the Q factor applies to the notches\n'
                                        'If ‘peak’, then the Q factor applies to the peaks')
            self.ftype_combx.addItems(['notch', 'peak'])
            self.ftype_combx.setCurrentIndex(0)

            self.pass_zero_checkbx = CheckBox('Pass Zero')
            self.pass_zero_checkbx.setToolTip('If False (default)\n'
                                              'the notches (nulls) of the filter are centered on frequencies [0, w0, 2*w0, …]\n'
                                              'and the peaks are centered on the midpoints [w0/2, 3*w0/2, 5*w0/2, …]\n'
                                              'If True, the peaks are centered on [0, w0, 2*w0, …] (passing zero frequency) and vice versa')
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
        try:
            if self.filter == iircomb:
                self.b, self.a = self.filter(w0=self.w0, Q=self.Q, ftype=self.ftype, fs=self.fs,
                                             pass_zero=self.pass_zero)
            else:
                self.b, self.a = self.filter(w0=self.w0, Q=self.Q, fs=self.fs)
        except Exception as err:
            printError(err)
