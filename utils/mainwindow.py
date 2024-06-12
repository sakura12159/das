# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/12 上午9:27
@Author  : zxy
@File    : mainwindow.py
"""
import ctypes
import os.path
import re
import sys
from itertools import cycle

import pandas as pd
import pywt
from PyEMD import EMD, EEMD, CEEMDAN
from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, qApp, QTabWidget, QTableWidget, QAbstractItemView, \
    QTableWidgetItem, QHeaderView, QTabBar, QScrollArea, QScrollBar, QVBoxLayout, QHBoxLayout
from image.image import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import integrate
from scipy.signal import hilbert, spectrogram
from scipy.signal.windows import *

from .classes.data_sifting import DataSifting
from .classes.filters import Filter
from .functions import *
from .widgets import *


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.initMainWindow()
        self.initGlobalParams()
        self.initUI()
        self.initMenu()
        self.initLayout()

    def initMainWindow(self):
        """获取屏幕分辨率，设置主窗口初始大小"""
        self.screen = QApplication.desktop()
        self.screen_height = int(self.screen.screenGeometry().height() * 0.8)
        self.screen_width = int(self.screen.screenGeometry().width() * 0.8)
        self.resize(self.screen_width, self.screen_height)

    def initGlobalParams(self):
        """初始化全局参数，即每次选择文件不改变"""
        # plt绘图参数
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # pg组件设置
        pg.setConfigOptions(leftButtonPan=True)  # 设置可用鼠标缩放
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')  # 设置界面前背景色

        # 输出设置
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)  # 设置输出时每行的长度

        self.channel_number = 1  # 当前通道
        self.channel_number_step = 1  # 通道号递增减步长
        self.files_read_number = 1  # 表格连续读取文件数
        self.files_read_number_changed = True  # 连续读取文件数是否改变

        # 滤波器是否更新数据
        self.update_data = False

        # 滤波器
        self.filter = None

        # 数据筛选
        self.data_sift = None

    def initUI(self):
        """初始化ui"""
        self.status_bar = self.statusBar()  # 状态栏
        self.status_bar.setStyleSheet('font-size: 15px; font-family: "Times New Roman", "SimHei";')
        self.menu_bar = self.menuBar()  # 菜单栏
        self.menu_bar.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "SimHei";')
        self.setWindowTitle('DAS数据查看')
        setPicture(self, icon_jpg, 'icon.jpg', window_icon=True)

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('myappid')  # 设置任务栏图标

    def initMenu(self):
        """初始化菜单"""
        # 文件
        self.file_menu = Menu('文件', self.menu_bar)

        # 导入
        self.import_action = Action('导入', self.file_menu, '导入数据文件', self.importData, shortcut='Ctrl+I')

        # 文件-导出
        self.export_action = Action('导出', self.file_menu, '导出数据', self.exportData, shortcut='Ctrl+E')

        self.file_menu.addSeparator()

        # 退出
        self.quit_action = Action('退出', self.file_menu, '退出软件', qApp.quit, shortcut='Ctrl+Q')

        # 操作
        self.operation_menu = Menu('操作', self.menu_bar, enabled=False)

        # 操作-计算信噪比
        self.calculate_snr_action = Action('计算信噪比', self.operation_menu, '计算选中数据的信噪比',
                                           self.calculateSNRDialog)

        self.operation_menu.addSeparator()

        # 操作-裁剪数据（时间）
        self.set_time_range_action = Action('查看范围（时间）', self.operation_menu,
                                            '按时间设置数据查看范围', self.setTimeRangeDialog)

        # 操作-裁剪数据（通道号）
        self.set_channel_range_action = Action('查看范围（通道）', self.operation_menu,
                                               '按通道设置数据查看范围', self.setChannelRangeDialog)

        self.operation_menu.addSeparator()

        # 操作-设置通道切换步长
        self.change_channel_number_step_action = Action('设置通道切换步长', self.operation_menu, '设置切换通道时的步长',
                                                        self.changeChannelNumberStep)

        # 操作-设置文件读取数量
        self.change_files_read_number_action = Action('设置文件读取数量', self.operation_menu,
                                                      '设置从表格选中文件时的读取数量，从选中的文件开始算起',
                                                      self.changeFilesReadNumberDialog)

        # 绘图
        self.plot_menu = Menu('绘图', self.menu_bar, enabled=False)

        # 绘图-时域特征
        self.plot_time_domain_features_menu = Menu('时域特征', self.plot_menu, status_tip='绘制所有通道的时域特征')

        # 绘图-时域特征-最大值等
        self.time_domain_chars_text = ['最大值', '峰值', '最小值', '平均值', '峰峰值', '绝对平均值', '均方根值',
                                       '方根幅值',
                                       '方差', '标准差', '峭度', '偏度', '裕度因子', '波形因子', '脉冲因子', '峰值因子',
                                       '峭度因子']
        for i in self.time_domain_chars_text:
            _ = Action(i, self.plot_time_domain_features_menu, f'绘制{i}图', self.plotTimeDomainFeature)

        # 绘图-二值图
        self.plot_binary_image_action = Action('二值图', self.plot_menu, '通过设置或计算阈值来绘制二值图',
                                               self.binaryImageDialog)

        # 绘图-热力图
        self.plot_heatmap_action = Action('热力图', self.plot_menu, '绘制热力图',
                                          lambda: self.plotFalseColorImage(type='heatmap'))

        # 绘图-多通道云图
        self.plot_multichannel_image_action = Action('多通道云图', self.plot_menu, '绘制多通道云图',
                                                     self.plotMultiWavesImage)

        # 绘图-应变图
        self.plot_strain_image_action = Action('应变图', self.plot_menu, '绘制应变图，单位为微应变', self.plotStrain)

        self.plot_menu.addSeparator()

        # 绘图-频域特征
        self.plot_freq_domain_features_menu = Menu('频域特征', self.plot_menu, status_tip='绘制所有通道的频域特征')

        # 绘图-频域特征-重心频率等
        self.fre_domain_chars_text = ['重心频率', '平均频率', '均方根频率', '均方频率', '频率方差', '频率标准差']
        for i in self.fre_domain_chars_text:
            _ = Action(i, self.plot_freq_domain_features_menu, f'绘制{i}图', self.plotFrequencyDomainFeature)

        # 绘图-fft
        self.plot_fft_menu = Menu('fft', self.plot_menu)

        # 绘图-fft-幅度谱
        self.plot_mag_spectrum_action = Action('幅度谱', self.plot_fft_menu, '绘制幅度谱',
                                               self.plotMagnitudeSpectrum)

        # 绘图-fft-角度谱
        self.plot_ang_spectrum_action = Action('角度谱', self.plot_fft_menu, '绘制角度谱', self.plotAngleSpectrum)

        # 绘图-fft-功率谱密度
        self.plot_psd_action = Action('功率谱密度', self.plot_fft_menu, '绘制功率谱密度', self.plotPSD)

        # 绘图-stft
        self.plot_stft_menu = Menu('stft', self.plot_menu)

        # 绘图-stft-功率谱密度
        self.plot_2d_psd_action = Action('功率谱密度', self.plot_stft_menu, '绘制功率谱密度', self.plot2dPSD)

        # 绘图-stft-三维功率谱密度
        self.plot_3d_psd_action = Action('三维功率谱密度', self.plot_stft_menu, '绘制三维功率谱密度', self.plot3dPSD)

        # 绘图-stft-幅度谱
        self.plot_2d_mag_spectrum_action = Action('幅度谱', self.plot_stft_menu, '绘制幅度谱',
                                                  self.plot2dMagnitudeSpectrum)

        # 绘图-stft-三维幅度谱
        self.plot_3d_mag_psd_action = Action('三维幅度谱', self.plot_stft_menu, '绘制三维幅度谱',
                                             self.plot3dMagnitudeSpectrum)

        # 绘图-stft-角度谱
        self.plot_2d_ang_spectrum_action = Action('角度谱', self.plot_stft_menu, '绘制角度谱',
                                                  self.plot2dAngleSpectrum)

        # 绘图-stft-三维角度谱
        self.plot_3d_ang_spectrum_action = Action('三维角度谱', self.plot_stft_menu, '绘制三维角度谱',
                                                  self.plot3dAngleSpectrum)

        self.plot_stft_menu.addSeparator()

        # 绘图-加窗设置
        self.window_options_action = Action('加窗设置', self.plot_stft_menu, '设置加窗参数', self.windowOptionsDialog)

        # 滤波
        self.filter_menu = Menu('滤波', self.menu_bar, enabled=False)

        # 滤波-更新数据
        self.update_data_action = Action('更新数据（否）', self.filter_menu, '如果为是，每次滤波后数据会更新',
                                         self.updateFilteredData)

        self.filter_menu.addSeparator()

        # 滤波-EMD
        self.emd_menu = Menu('EMD', self.filter_menu, status_tip='使用EMD及衍生方式滤波')

        # 滤波-EMD-EMD, EEMD and CEEMDAN
        emd_method_list = ['EMD', 'EEMD', 'CEEMDAN']
        for i in emd_method_list:
            _ = Action(i, self.emd_menu, f'使用{i}进行数据分解与重构', self.plotEMD)

        self.emd_menu.addSeparator()

        # 滤波-EMD-绘制瞬时频率
        self.emd_plot_ins_fre_action = Action('绘制瞬时频率', self.emd_menu, '绘制重构IMF的瞬时频率',
                                              self.plotEMDInstantaneousFrequency)
        self.emd_plot_ins_fre_action.setEnabled(False)

        self.emd_menu.addSeparator()

        # 滤波-EMD-设置
        self.emd_options_action = Action('设置', self.emd_menu, 'EMD相关设置', self.EMDOptionsDialog)

        # 滤波-IIR滤波器
        self.iir_menu = Menu('IIR滤波器', self.filter_menu)

        # 滤波-IIR滤波器-Butterworth等
        cal_filter_types = ['Butterworth', 'Chebyshev type I', 'Chebyshev type II', 'Elliptic (Cauer)']
        for i in cal_filter_types:
            _ = Action(i, self.iir_menu, f'设计{i}滤波器', self.iirFilterDesign)

        self.iir_menu.addSeparator()

        # 滤波-IIR滤波器-Bessel/Thomson
        self.iir_bessel_action = Action('Bessel/Thomson', self.iir_menu, '设计Bessel/Thomson滤波器',
                                        self.iirFilterDesign)

        self.iir_menu.addSeparator()

        # 滤波-IIR滤波器-notch等
        comb_filter_types = ['Notch Digital Filter', 'Peak (Resonant) Digital Filter',
                             'Notching or Peaking Digital Comb Filter']
        for i in comb_filter_types:
            _ = Action(i, self.iir_menu, f'设计{i}滤波器', self.iirFilterDesign)

        # 滤波-小波
        self.wavelet_menu = Menu('小波', self.filter_menu)

        # 滤波-小波-离散小波变换
        self.wavelet_dwt_action = Action('离散小波变换', self.wavelet_menu, '使用离散小波变换进行数据分解与重构或去噪',
                                         self.waveletDWTDialog)

        # 滤波-小波-小波去噪
        self.wavelet_threshold_action = Action('小波去噪', self.wavelet_menu, '小波去噪', self.waveletThresholdDialog)

        self.wavelet_menu.addSeparator()

        # 滤波-小波-小波包
        self.wavelet_packets_action = Action('小波包', self.wavelet_menu, '使用小波包进行数据分解并从选择的节点重构',
                                             self.waveletPacketsDialog)

        # 其他
        self.others_menu = Menu('其他', self.menu_bar, enabled=True)

        # 其他-数据筛选
        self.data_sifting_action = Action('数据筛选', self.others_menu, '使用双门限法筛选数据',
                                          self.dataSiftingDialog)

    def initLayout(self):
        """初始化主窗口布局"""
        # 设置主窗口layout
        main_window_widget = QWidget()
        main_window_vbox = QVBoxLayout()
        main_window_hbox = QHBoxLayout()

        # 左侧
        # 文件区
        file_hbox = QHBoxLayout()
        file_area_vbox = QVBoxLayout()
        file_path_label = Label('文件路径')
        self.file_path_line_edit = LineEdit(focus=False)

        change_file_path_button = PushButton()
        setPicture(change_file_path_button, folder_jpg, 'folder.jpg', )
        change_file_path_button.clicked.connect(self.changeFilePath)

        file_table_scrollbar = QScrollBar(Qt.Vertical)
        file_table_scrollbar.setStyleSheet('min-height: 100')  # 设置滚动滑块的最小高度
        self.files_table_widget = QTableWidget(100, 1)
        self.files_table_widget.setVerticalScrollBar(file_table_scrollbar)
        self.files_table_widget.setStyleSheet('font-size: 17px; font-family: "Times New Roman", "Microsoft YaHei";')
        self.files_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.files_table_widget.setHorizontalHeaderLabels(['文件'])  # 设置表头
        self.files_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        QTableWidget.resizeRowsToContents(self.files_table_widget)
        QTableWidget.resizeColumnsToContents(self.files_table_widget)  # 设置表格排与列的宽度随内容改变
        self.files_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置一次选中一排内容
        self.files_table_widget.currentItemChanged.connect(self.selectDataFromTable)

        # 文件区布局
        file_hbox.addWidget(self.file_path_line_edit)
        file_hbox.addWidget(change_file_path_button)
        file_area_vbox.addWidget(file_path_label)
        file_area_vbox.addLayout(file_hbox)
        file_area_vbox.addWidget(self.files_table_widget)

        # 右侧
        # 参数
        sampling_rate_label = Label('采样率')
        self.sampling_rate_line_edit = LineEditWithReg(focus=False)

        sampling_times_label = Label('采样次数')
        self.current_sampling_times_line_edit = LineEditWithReg(focus=False)

        number_of_channels_label = Label('通道数')
        self.current_channels_line_edit = LineEditWithReg(focus=False)

        channel_number_label = Label('通道号')
        self.channel_number_spinbx = SpinBox()
        self.channel_number_spinbx.setValue(1)
        self.channel_number_spinbx.setMinimumWidth(100)
        self.channel_number_spinbx.valueChanged.connect(self.changeChannelNumber)
        self.channel_number_spinbx.valueChanged.connect(self.plotSingleChannelTime)
        self.channel_number_spinbx.valueChanged.connect(self.plotAmplitudeFrequency)

        # 播放音频按钮
        self.playBtn = PushButton()
        setPicture(self.playBtn, play_jpg, 'play.jpg')
        self.playBtn.clicked.connect(self.createWavFile)
        self.playBtn.clicked.connect(self.createPlayer)
        self.playBtn.clicked.connect(self.playBtnChangeState)

        # 停止音频播放按钮
        self.stopBtn = PushButton()
        setPicture(self.stopBtn, stop_jpg, 'stop.jpg')
        self.stopBtn.clicked.connect(self.resetPlayer)

        self.playBtn.setDisabled(True)
        self.stopBtn.setDisabled(True)  # 默认不可选中

        # 数据参数布局
        data_params_hbox = QHBoxLayout()
        data_params_hbox.addWidget(channel_number_label)
        data_params_hbox.addWidget(self.channel_number_spinbx)
        data_params_hbox.addSpacing(20)
        data_params_hbox.addWidget(self.playBtn)
        data_params_hbox.addSpacing(5)
        data_params_hbox.addWidget(self.stopBtn)
        data_params_hbox.addSpacing(20)
        data_params_hbox.addWidget(sampling_rate_label)
        data_params_hbox.addWidget(self.sampling_rate_line_edit)
        data_params_hbox.addSpacing(5)
        data_params_hbox.addWidget(sampling_times_label)
        data_params_hbox.addWidget(self.current_sampling_times_line_edit)
        data_params_hbox.addSpacing(5)
        data_params_hbox.addWidget(number_of_channels_label)
        data_params_hbox.addWidget(self.current_channels_line_edit)

        # 绘制灰度图
        self.plot_gray_scale_widget = MyPlotWidget('灰度图', '时间（s）', '通道', check_mouse=False)

        # 绘制单通道相位差-时间图
        self.plot_single_channel_time_widget = MyPlotWidget('相位差图', '时间（s）', '相位差（rad）', grid=True)

        # 绘制频谱图
        self.plot_amplitude_frequency_widget = MyPlotWidget('幅值图', '频率（Hz）', '幅值', grid=True)

        combine_image_widget = QWidget()
        image_vbox = QVBoxLayout()
        image_vbox.addWidget(self.plot_single_channel_time_widget)
        image_vbox.addWidget(self.plot_amplitude_frequency_widget)
        combine_image_widget.setLayout(image_vbox)

        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)  # 设置tab可移动
        self.tab_widget.setStyleSheet('font-size: 15px; font-family: "Times New Roman", "Microsoft YaHei";')
        self.tab_widget.setTabsClosable(True)  # 设置tab可关闭
        self.tab_widget.tabCloseRequested[int].connect(self.removeTab)
        self.tab_widget.addTab(self.plot_gray_scale_widget, '灰度图')
        self.tab_widget.addTab(combine_image_widget, '单通道')
        self.tab_widget.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tab_widget.tabBar().setTabButton(1, QTabBar.RightSide, None)  # 设置删除按钮消失

        # GPS时间组件
        gps_from_label = Label('始')
        gps_to_label = Label('止')
        self.gps_from_line_edit = LineEdit(focus=False)
        self.gps_to_line_edit = LineEdit(focus=False)

        # GPS时间布局
        gps_hbox = QHBoxLayout()
        gps_hbox.addWidget(gps_from_label)
        gps_hbox.addWidget(self.gps_from_line_edit)
        gps_hbox.addSpacing(50)
        gps_hbox.addWidget(gps_to_label)
        gps_hbox.addWidget(self.gps_to_line_edit)

        # 右侧布局
        main_window_vbox.addSpacing(10)
        main_window_vbox.addLayout(data_params_hbox)
        main_window_vbox.addSpacing(10)
        main_window_vbox.addWidget(self.tab_widget)
        main_window_vbox.addSpacing(10)
        main_window_vbox.addLayout(gps_hbox)

        # 主页面
        main_window_hbox.addSpacing(10)
        main_window_hbox.addLayout(file_area_vbox)
        main_window_hbox.addSpacing(20)
        main_window_hbox.addLayout(main_window_vbox)
        main_window_hbox.addSpacing(10)
        main_window_hbox.setStretchFactor(file_area_vbox, 1)
        main_window_hbox.setStretchFactor(main_window_vbox, 4)  # 设置各部分所占比例
        main_window_widget.setLayout(main_window_hbox)
        self.setCentralWidget(main_window_widget)

    def initLocalParams(self):
        """初始化局部参数，即适用于单个文件，重新选择文件后更新"""
        # 默认错误为None
        self.err = None  # 捕捉到的错误

        # 播放器默认状态
        if hasattr(self, 'player'):
            self.player.stop()
        setPicture(self.playBtn, play_jpg, 'play.jpg')
        self.playBtn.setDisabled(False)
        self.stopBtn.setDisabled(False)  # 设置播放按钮
        self.playerState = False  # 播放器否在播放
        self.hasWavFile = False  # 当前通道是否已创建了音频文件
        self.playerHasMedia = False  # 播放器是否已赋予了文件

        if self.files_read_number_changed:
            self.channel_from_num = 1
            self.channel_to_num = self.channels_num
            self.sampling_times_from_num = 1
            self.sampling_times_to_num = self.sampling_times
            self.files_read_number_changed = False

        # 信噪比
        self.signal_channel_number = 1
        self.noise_channel_number = 1
        self.signal_start_sampling_time = 1
        self.signal_stop_sampling_time = 2
        self.noise_start_sampling_time = 1
        self.noise_stop_sampling_time = 2

        # 二值图
        self.binary_image_flag = True  # 是否使用简单阈值
        self.binary_image_threshold = 120.0  # 阈值
        self.binary_image_threshold_methods = {'双峰法': twoPeaks, '大津法': ostu}  # 两种计算方法，双峰法及大津法
        self.binary_image_threshold_method_index = 0  # 计算阈值方法的索引

        # 加窗
        self.window_length = 256  # 窗长
        self.window_text = 'Rectangular / Dirichlet'  # 加窗名称
        self.window_method = boxcar  # 加窗种类
        self.window_overlap_size_ratio = 0.75  # 窗口重叠比
        self.window_overlap_size = int(round(self.window_overlap_size_ratio * self.window_length))  # 默认窗口重叠长度，取整
        self.window_methods = {'Bartlett': bartlett, 'Blackman': blackman, 'Blackman-Harris': blackmanharris,
                               'Bohman': bohman, 'Cosine': cosine, 'Flat Top': flattop,
                               'Hamming': hamming, 'Hann': hann, 'Lanczos / Sinc': lanczos,
                               'Modified Barrtlett-Hann': barthann, 'Nuttall': nuttall, 'Parzen': parzen,
                               'Rectangular / Dirichlet': boxcar, 'Taylor': taylor, 'Triangular': triang,
                               'Tukey / Tapered Cosine': tukey}  # 默认窗口名称及对应的窗口

        # EMD
        self.imfs_res_num = 5  # 所有模态加残余模态的数量
        self.reconstruct_nums = str([i for i in range(1, self.imfs_res_num)])  # 重构模态号
        self.emd_options_flag = True  # 默认操作为分解
        self.eemd_trials = 100  # 添加噪声点数量？
        self.eemd_noise_width = 0.05  # 添加的高斯噪声的标准差
        self.ceemdan_trials = 100  # 添加噪声点数量？
        self.ceemdan_epsilon = 0.005  # 添加噪声大小与标准差的乘积
        self.ceemdan_noise_scale = 1.0  # 添加噪声的大小
        self.ceemdan_noise_kind_index = 0  # 添加噪声种类的索引
        self.ceemdan_range_thr = 0.01  # 范围阈值，小于则不再分解
        self.ceemdan_total_power_thr = 0.05  # 总功率阈值，小于则不再分解

        # 小波分解
        self.wavelet_dwt_flag = True  # 默认操作为分解
        self.wavelet_dwt_reconstruct = ['cA1', 'cD1']  # 重构系数名称
        self.wavelet_dwt_family_index = 0  # 小波族索引
        self.wavelet_dwt_name_index = 0  # 小波索引，默认显示第一个
        self.wavelet_dwt_decompose_level = 1  # 分解层数
        self.wavelet_dwt_decompose_level_calculated = False  # 是否使用函数计算最大分解层数
        self.wavelet_dwt_padding_mode_index = 0  # 数据填充模式索引

        # 小波去噪
        self.wavelet_threshold = 1.0  # 阈值
        self.wavelet_threshold_sub = 0.0  # 替换值
        self.wavelet_threshold_modes = ['soft', 'hard', 'garrote', 'greater', 'less']  # 阈值种类
        self.wavelet_threshold_mode_index = 0  # 阈值种类索引

        # 小波包分解
        self.wavelet_packets_flag = True  # 默认操作为分解
        self.wavelet_packets_reconstruct = ['a', 'd']  # 重构节点名称
        self.wavelet_packets_family_index = 0  # 小波族索引
        self.wavelet_packets_name_index = 0  # 小波索引
        self.wavelet_packets_decompose_level = 1  # 分解层数
        self.wavelet_packets_decompose_max_level = None  # 显示的最大分解层数
        self.wavelet_packets_padding_mode_index = 0  # 数据填充模式索引

    # """------------------------------------------------------------------------------------------------------------"""
    """继承mainwindow自带函数"""

    # def closeEvent(self, event):
    #     """退出时的提示"""
    #     reply = QMessageBox.question(self, '提示', '是否退出？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #
    #     # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
    #     event.accept() if reply == QMessageBox.Yes else event.ignore()

    def removeTab(self, index: int) -> None:
        """关闭对应选项卡"""
        self.tab_widget.removeTab(index)

    # """------------------------------------------------------------------------------------------------------------"""
    """播放当前文件调用的函数"""

    def playBtnChangeState(self):
        """点击播放按钮改变文字和播放器状态"""
        if not self.playerState:
            setPicture(self.playBtn, pause_jpg, 'pause.jpg')
            self.player.play()
            self.playerState = True
        else:
            setPicture(self.playBtn, play_jpg, 'play.jpg')
            self.player.pause()
            self.playerState = False

    def createWavFile(self):
        """创建当前数据的wav文件，储存在当前文件夹路径下"""
        if not self.hasWavFile:
            data = np.array(self.data[self.channel_number - 1])  # 不转array会在重复转换数据类型时发生数据类型错误
            writeWav(self.file_path, data, self.sampling_rate)
            self.hasWavFile = True

    def createPlayer(self):
        """创建播放器并赋数据"""
        if not self.playerHasMedia:
            self.player = QtMultimedia.QMediaPlayer()
            self.player.stateChanged.connect(self.playerStateChanged)
            self.player.setMedia(
                QtMultimedia.QMediaContent(QUrl.fromLocalFile(os.path.join(self.file_path, 'temp.wav'))))
            self.playerHasMedia = True

    def playerStateChanged(self, state: QtMultimedia.QMediaPlayer.State) -> None:
        """播放器停止后删除文件"""
        if state == QtMultimedia.QMediaPlayer.StoppedState:
            self.resetPlayer()
            os.remove(os.path.join(self.file_path, 'temp.wav'))  # 在播放完成或点击Abort后删除临时文件

    def resetPlayer(self):
        """重置播放器"""
        self.player.stop()
        setPicture(self.playBtn, play_jpg, 'play.jpg')
        self.playerState = False
        self.hasWavFile = False
        self.playerHasMedia = False

    # """------------------------------------------------------------------------------------------------------------"""
    """主绘图区"""

    def plotSingleChannelTime(self):
        """绘制单通道相位差-时间图"""
        self.plot_single_channel_time_widget.plot_item.clear()

        x = self.xAxis()
        data = self.data[self.channel_number - 1]
        self.plot_single_channel_time_widget.draw(x, data, pen=QColor('blue'))

    def plotAmplitudeFrequency(self):
        """绘制幅值-频率图"""
        self.plot_amplitude_frequency_widget.plot_item.clear()

        data = self.data[self.channel_number - 1]
        data = toAmplitude(data)
        x = self.xAxis(freq=True)

        self.plot_amplitude_frequency_widget.draw(x, data, pen=QColor('blue'))

    # """------------------------------------------------------------------------------------------------------------"""
    """文件路径区和文件列表调用函数"""

    def changeFilePath(self):
        """更改显示的文件路径"""
        file_path = QFileDialog.getExistingDirectory(self, '设置文件路径', '')  # 起始路径
        if file_path != '':
            self.file_path = file_path
            self.updateFile()

    def selectDataFromTable(self):
        """当从文件列表中选择文件时更新图像等"""
        self.file_names = []
        item_index = self.files_table_widget.currentIndex().row()  # 获取当前点击的文件行索引
        for i in range(self.files_read_number):
            if item_index + i + 1 > self.files_table_widget.rowCount():  # 如果读取文件数大于该文件下面剩余的文件数就只读到最后一个文件
                break
            self.file_names.append(self.files_table_widget.item(item_index + i, 0).text())

        self.readData()
        self.initLocalParams()
        self.updateAll()

    def changeChannelNumber(self):
        """更改通道号，默认为1"""
        self.channel_number = 1 if self.channel_number_spinbx.value() == '' else self.channel_number_spinbx.value()

    # """------------------------------------------------------------------------------------------------------------"""
    """更新函数"""

    def updateMenuBar(self):
        """更新菜单可操作性状态"""
        self.export_action.setEnabled(True)
        self.operation_menu.setEnabled(True)
        self.plot_menu.setEnabled(True)
        self.filter_menu.setEnabled(True)

    def updateFile(self):
        """更新文件列表显示"""
        self.file_path_line_edit.setText(self.file_path)
        files = [f for f in os.listdir(self.file_path) if f.endswith('.dat')]
        self.files_table_widget.setRowCount(len(files))  # 有多少个文件就显示多少行
        for i in range(len(files)):
            table_widget_item = QTableWidgetItem(files[i])
            self.files_table_widget.setItem(i, 0, table_widget_item)

    def updateDataRange(self):
        """更新数据显示范围"""
        self.data = self.origin_data[self.channel_from_num - 1:self.channel_to_num,
                    self.sampling_times_from_num - 1:self.sampling_times_to_num]

    def updateDataParams(self):
        """更新数据相关参数"""
        self.current_channels = self.channel_to_num - self.channel_from_num + 1
        self.current_sampling_times = self.sampling_times_to_num - self.sampling_times_from_num + 1

        self.channel_number_spinbx.setRange(1, self.current_channels)
        self.channel_number_spinbx.setValue(self.channel_number)
        self.sampling_rate_line_edit.setText(str(self.sampling_rate))
        self.current_sampling_times_line_edit.setText(str(self.current_sampling_times))
        self.current_channels_line_edit.setText(str(self.current_channels))

    def updateDataGPSTime(self):
        """更新数据时间显示"""
        from_time, to_time = [], []
        for i in range(6):
            ftime = str(self.time[0][i]) if i == 5 else str(self.time[0][i])[:-2]
            ttime = str(self.time[-1][i]) if i == 5 else str(self.time[-1][i])[:-2]
            from_time.append(ftime)
            to_time.append(ttime)
        from_time = ' - '.join(from_time)
        to_time = ' - '.join(to_time)

        self.gps_from_line_edit.setText(from_time)  # 更新开头文件GPS时间
        self.gps_to_line_edit.setText(to_time)  # 更新末尾文件GPS时间

    def updateImages(self):
        """更新4个随时更新的图像显示"""
        self.plotFalseColorImage(type='gray')
        self.plotSingleChannelTime()
        self.plotAmplitudeFrequency()

    def updateAll(self):
        """总更新函数"""
        self.updateMenuBar()
        self.updateFile()
        self.updateDataRange()
        self.updateDataParams()
        self.updateDataGPSTime()
        self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""
    """功能函数"""

    def xAxis(self, begin: int = None, end: int = None, num: int = None, freq: bool = False) -> np.array:
        """生成绘制时域图的x轴"""
        if not num:
            num = self.current_sampling_times

        if not freq:
            if not begin:
                begin = self.sampling_times_from_num
            if not end:
                end = self.sampling_times_to_num
            return np.linspace(begin, end, num) / self.sampling_rate
        else:
            return np.fft.fftfreq(num, 1 / self.sampling_rate)[:num // 2]

    def initTwoPlotWidgets(self, data: np.array, title: str) -> QWidget:
        """创建返回结合两个pw的Qwidget"""
        x = self.xAxis()
        data_widget = MyPlotWidget(f'{title}', '时间（s）', '相位差（rad）', grid=True)
        data_widget.draw(x, data, pen=QColor('blue'))

        data = toAmplitude(data)
        x = self.xAxis(freq=True)
        fre_amp_widget = MyPlotWidget('幅值图', '频率（Hz）', '幅值', grid=True)
        fre_amp_widget.draw(x, data, pen=QColor('blue'))

        combine_widget = QWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(data_widget)
        vbox.addWidget(fre_amp_widget)
        combine_widget.setLayout(vbox)
        return combine_widget

    def initFalseColorWidget(self, data: np.array, plot_widget: MyPlotWidget, type: str = None) -> None:
        """初始化伪颜色图widget"""
        if type == 'gray':
            plot_widget.clear()

        tr = QTransform()
        tr.scale(1 / self.sampling_rate, 1)  # 缩放
        tr.translate(self.sampling_times_from_num, 0)  # 移动

        item = pg.ImageItem()
        if type == 'heatmap':
            item.setColorMap('viridis')
        item.setImage(data.T)
        item.setTransform(tr)
        plot_widget.addItem(item)

    # """------------------------------------------------------------------------------------------------------------"""
    """文件-导入菜单调用函数"""

    def importData(self):
        """导入（多个）数据文件后更新参数和绘图等"""
        file_names = QFileDialog.getOpenFileNames(self, '导入', '', 'DAS data (*.dat)')[0]  # 打开多个.dat文件
        if file_names:
            self.file_names = file_names
            self.file_path = os.path.dirname(self.file_names[0])

            self.readData()
            self.initLocalParams()
            self.updateAll()

    def readData(self):
        """读取数据，更新参数"""
        time, data = [], []
        for file in self.file_names:
            raw_data = np.fromfile(os.path.join(self.file_path, file), dtype='<f4')  # <低位在前高位在后（小端模式），f4：32位（单精度）浮点类型
            self.sampling_rate = int(raw_data[6])  # 采样率
            self.single_sampling_times = int(raw_data[7])  # 采样次数
            self.channels_num = int(raw_data[9])  # 通道数
            time.append(raw_data[:6])  # GPS时间
            data.append(raw_data[10:].reshape(self.channels_num, self.single_sampling_times))

        self.time = time
        self.data = np.concatenate(data, axis=1)  # （通道数，采样次数）
        # self.data = filterData(self.data)
        sampling_times = self.single_sampling_times * len(self.file_names)
        if self.files_read_number_changed:
            self.sampling_times = sampling_times
        else:
            if self.sampling_times != sampling_times:
                self.sampling_times = sampling_times
                self.files_read_number_changed = True

        self.origin_data = self.data

    # """------------------------------------------------------------------------------------------------------------"""
    """文件-导出调用函数"""

    def exportData(self):
        """导出数据"""
        fpath, ftype = QFileDialog.getSaveFileName(self, '导出', '', 'csv(*.csv);;json(*.json);;pickle(*.pickle);;'
                                                                     'txt(*.txt);;xls(*.xls *.xlsx)')

        data = pd.DataFrame(self.data)  # 保存为df

        if ftype.find('*.txt') > 0:  # txt以空格分隔
            data.to_csv(fpath, sep=' ', index=False, header=False)
        elif ftype.find('*.csv') > 0:  # csv以逗号分隔
            data.to_csv(fpath, sep=',', index=False, header=False)
        elif ftype.find('*.xls') > 0:  # xls以制表符分隔
            data.to_csv(fpath, sep='\t', index=False, header=False)
        elif ftype.find('*.json') > 0:
            data.to_json(fpath, orient='values')
        elif ftype.find('*.pickle') > 0:
            data.to_pickle(fpath)

    # """------------------------------------------------------------------------------------------------------------"""
    """计算信噪比调用函数"""

    def calculateSNRDialog(self):
        """计算信噪比对话框"""
        self.dialog = Dialog()
        self.dialog.setMaximumWidth(500)
        self.dialog.setWindowTitle('计算信噪比')

        signal_channel_number_label = Label('信号所在通道号')
        self.signal_channel_number_line_edit = LineEditWithReg()
        self.signal_channel_number_line_edit.setText(str(self.signal_channel_number))
        signal_start_sampling_time_label = Label('起始采样次数')
        self.signal_start_sampling_time_line_edit = LineEditWithReg()
        self.signal_start_sampling_time_line_edit.setText(str(self.signal_start_sampling_time))
        signal_stop_sampling_time_label = Label('终止采样次数')
        self.signal_stop_sampling_time_line_edit = LineEditWithReg()
        self.signal_stop_sampling_time_line_edit.setText(str(self.signal_stop_sampling_time))

        noise_channel_number_label = Label('噪声所在通道号')
        self.noise_channel_number_line_edit = LineEditWithReg()
        self.noise_channel_number_line_edit.setText(str(self.noise_channel_number))
        noise_start_sampling_time_label = Label('起始采样次数')
        self.noise_start_sampling_time_line_edit = LineEditWithReg()
        self.noise_start_sampling_time_line_edit.setText(str(self.noise_start_sampling_time))
        noise_stop_sampling_time_label = Label('终止采样次数')
        self.noise_stop_sampling_time_line_edit = LineEditWithReg()
        self.noise_stop_sampling_time_line_edit.setText(str(self.noise_stop_sampling_time))

        snr_label = Label('SNR = ')
        self.snr_line_edit = LineEditWithReg(focus=False)
        self.snr_line_edit.setMaximumWidth(100)
        snr_unit_label = Label('dB')

        btn = PushButton('计算')
        btn.clicked.connect(self.updateCalculateSNRParams)
        btn.clicked.connect(self.calculateSNR)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()

        hbox1.addWidget(signal_channel_number_label)
        hbox1.addWidget(self.signal_channel_number_line_edit)
        hbox1.addSpacing(50)
        hbox1.addWidget(signal_start_sampling_time_label)
        hbox1.addWidget(self.signal_start_sampling_time_line_edit)
        hbox1.addSpacing(5)
        hbox1.addWidget(signal_stop_sampling_time_label)
        hbox1.addWidget(self.signal_stop_sampling_time_line_edit)

        hbox2.addWidget(noise_channel_number_label)
        hbox2.addWidget(self.noise_channel_number_line_edit)
        hbox2.addSpacing(50)
        hbox2.addWidget(noise_start_sampling_time_label)
        hbox2.addWidget(self.noise_start_sampling_time_line_edit)
        hbox2.addSpacing(5)
        hbox2.addWidget(noise_stop_sampling_time_label)
        hbox2.addWidget(self.noise_stop_sampling_time_line_edit)

        hbox3.addStretch(1)
        hbox3.addWidget(snr_label)
        hbox3.addWidget(self.snr_line_edit)
        hbox3.addWidget(snr_unit_label)
        hbox3.addStretch(1)

        vbox.addLayout(hbox1)
        vbox.addSpacing(5)
        vbox.addLayout(hbox2)
        vbox.addSpacing(5)
        vbox.addLayout(hbox3)
        vbox.addSpacing(5)
        vbox.addWidget(btn)

        self.dialog.setLayout(vbox)
        self.dialog.exec_()

    def updateCalculateSNRParams(self):
        """更新参数"""
        self.signal_channel_number = int(self.signal_channel_number_line_edit.text())

        self.noise_channel_number = int(self.noise_channel_number_line_edit.text())

        self.signal_start_sampling_time = int(self.signal_stop_sampling_time_line_edit.text())

        self.signal_stop_sampling_time = int(self.signal_stop_sampling_time_line_edit.text())

        self.noise_start_sampling_time = int(self.noise_start_sampling_time_line_edit.text())

        self.noise_stop_sampling_time = int(self.noise_stop_sampling_time_line_edit.text())

    def calculateSNR(self):
        """计算信噪比"""
        try:
            signal_data = self.data[self.signal_channel_number,
                          self.signal_start_sampling_time:self.signal_stop_sampling_time + 1]
            noise_data = self.data[self.noise_channel_number,
                         self.noise_start_sampling_time:self.noise_stop_sampling_time + 1]

            snr = round(10.0 * np.log10(np.sum(signal_data ** 2) / np.sum(noise_data ** 2)), 5)
            self.snr_line_edit.setText(str(snr))
        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """操作-查看数据（时间）调用函数"""

    def setTimeRangeDialog(self):
        """调用按时间查看数据范围的对话框"""
        dialog = Dialog()
        dialog.setWindowTitle('查看数据（时间）')

        from_label = Label('始')
        self.time_range_from_line_edit = LineEditWithReg(digit=True)
        self.time_range_from_line_edit.setText(str(self.sampling_times_from_num / self.sampling_rate))
        to_label = Label('止')
        self.time_range_to_line_edit = LineEditWithReg(digit=True)
        self.time_range_to_line_edit.setText(str(self.sampling_times_to_num / self.sampling_rate))

        btn = PushButton('确定')
        btn.clicked.connect(self.setTimeRange)
        btn.clicked.connect(self.updateDataParams)
        btn.clicked.connect(self.updateImages)
        btn.clicked.connect(dialog.close)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(from_label)
        hbox.addWidget(self.time_range_from_line_edit)
        hbox.addStretch(1)
        hbox.addWidget(to_label)
        hbox.addWidget(self.time_range_to_line_edit)
        vbox.addLayout(hbox)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def setTimeRange(self):
        """根据设置时间截取数据"""
        from_num = int(float(self.time_range_from_line_edit.text()) * self.sampling_rate)
        to_num = int(float(self.time_range_to_line_edit.text()) * self.sampling_rate)

        if 1 <= from_num < self.origin_data.shape[1] and 1 < to_num <= self.origin_data.shape[1]:
            if from_num > to_num:
                from_num, to_num = to_num, from_num
        elif from_num == 0 and 1 < to_num <= self.origin_data.shape[1]:
            from_num = 1
        else:
            from_num, to_num = 1, self.origin_data.shape[1]

        self.sampling_times_from_num, self.sampling_times_to_num = from_num, to_num

        # 捕获索引错误等
        try:
            self.updateDataRange()
        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """查看数据（通道）调用函数"""

    def setChannelRangeDialog(self):
        """按通道截取数据的对话框"""
        dialog = Dialog()
        dialog.setWindowTitle('查看数据（通道）')

        from_label = Label('始')
        self.channel_from = LineEditWithReg()
        self.channel_from.setText(str(self.channel_from_num))
        to_label = Label('止')
        self.channel_to = LineEditWithReg()
        self.channel_to.setText(str(self.channel_to_num))

        btn = PushButton('确定')
        btn.clicked.connect(self.setChannelRange)
        btn.clicked.connect(self.updateDataParams)
        btn.clicked.connect(self.updateImages)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(from_label)
        hbox.addWidget(self.channel_from)
        hbox.addStretch(1)
        hbox.addWidget(to_label)
        hbox.addWidget(self.channel_to)

        vbox.addLayout(hbox)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def setChannelRange(self):
        """以通道数截取"""
        from_num, to_num = int(self.channel_from.text()), int(self.channel_to.text())

        if 1 <= from_num < self.origin_data.shape[0] and 1 < to_num <= self.origin_data.shape[0]:
            if from_num > to_num:
                from_num, to_num = to_num, from_num
        elif from_num == 0 and 1 < to_num <= self.origin_data.shape[0]:
            from_num = 1
        else:
            from_num, to_num = 1, self.origin_data.shape[0]

        self.channel_from_num, self.channel_to_num = from_num, to_num

        # 捕获索引错误等
        try:
            self.updateDataRange()
        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """更改读取通道号步长、读取文件数调用的函数"""

    def changeChannelNumberStep(self):
        """改变通道号的步长"""
        dialog = Dialog()
        dialog.setWindowTitle('设置通道切换步长')

        channel_number_step_label = Label('步长')
        self.channel_number_step_line_edit = LineEditWithReg()
        self.channel_number_step_line_edit.setToolTip('切换通道时的步长')
        self.channel_number_step_line_edit.setText(str(self.channel_number_step))

        btn = PushButton('确定')
        btn.clicked.connect(self.updateChannelNumberStep)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(channel_number_step_label)
        hbox.addWidget(self.channel_number_step_line_edit)
        vbox.addLayout(hbox)
        vbox.addSpacing(5)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateChannelNumberStep(self):
        """更新读取通道号的步长"""
        self.channel_number_step = int(self.channel_number_step_line_edit.text())

        self.channel_number_spinbx.setSingleStep(self.channel_number_step)

    def changeFilesReadNumberDialog(self):
        """从表格选择文件时读取的文件数"""
        dialog = Dialog()
        dialog.setWindowTitle('设置文件读取数量')

        files_read_number_label = Label('文件读取数量')
        self.files_read_number_line_edit = LineEditWithReg()
        self.files_read_number_line_edit.setToolTip('设置从表格选中文件时的读取数量，从选中的文件开始算起')
        self.files_read_number_line_edit.setText(str(self.files_read_number))

        btn = PushButton('确定')
        btn.clicked.connect(self.updateFilesReadNumber)
        btn.clicked.connect(self.updateImages)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(files_read_number_label)
        hbox.addWidget(self.files_read_number_line_edit)
        vbox.addLayout(hbox)
        vbox.addSpacing(5)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateFilesReadNumber(self):
        """更新读取文件数量"""
        files_read_number = int(self.files_read_number_line_edit.text())
        if self.files_read_number != files_read_number:
            self.files_read_number = files_read_number
            self.files_read_number_changed = True

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制热力图调用函数"""

    def plotFalseColorImage(self, type: str = 'gray') -> None:
        """绘制伪颜色图"""
        if type == 'gray':
            self.initFalseColorWidget(self.data, self.plot_gray_scale_widget, type='gray')

        elif type == 'binary':
            plot_widget = MyPlotWidget('二值图', '时间（s）', '通道', check_mouse=False)
            self.tab_widget.addTab(plot_widget, f'二值图 - 阈值={self.binary_image_threshold}')
            # 阈值化
            data = normalizeToGrayScale(self.data)
            data[data >= self.binary_image_threshold] = 255
            data[data < self.binary_image_threshold] = 0  # 根据阈值赋值
            self.initFalseColorWidget(data, plot_widget)

        elif type == 'heatmap':
            plot_widget = MyPlotWidget('热力图', '时间（s）', '通道', check_mouse=False)
            self.tab_widget.addTab(plot_widget, '热力图')
            self.initFalseColorWidget(self.data, plot_widget, type='heatmap')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制二值图调用函数"""

    def binaryImageDialog(self):
        """二值图设置组件"""
        dialog = Dialog()
        dialog.setWindowTitle('二值图')

        self.binary_image_input_radiobtn = RadioButton('阈值')
        self.binary_image_input_radiobtn.setChecked(self.binary_image_flag)

        self.binary_image_threshold_line_edit = LineEditWithReg(digit=True)
        self.binary_image_threshold_line_edit.setText(str(self.binary_image_threshold))

        self.binary_image_method_radiobtn = RadioButton('计算方法')
        self.binary_image_method_radiobtn.setChecked(not self.binary_image_flag)

        self.binary_image_method_combx = ComboBox()
        self.binary_image_method_combx.addItems(self.binary_image_threshold_methods.keys())
        self.binary_image_method_combx.setCurrentIndex(self.binary_image_threshold_method_index)

        btn = PushButton('确定')
        btn.clicked.connect(self.updateBinaryImageParams)
        btn.clicked.connect(lambda: self.plotFalseColorImage(type='binary'))
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.binary_image_input_radiobtn)
        hbox1.addWidget(self.binary_image_threshold_line_edit)
        hbox1.addStretch(0)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.binary_image_method_radiobtn)
        hbox2.addSpacing(20)
        hbox2.addWidget(self.binary_image_method_combx)
        hbox2.addStretch(0)

        vbox.addLayout(hbox1)
        vbox.addSpacing(30)
        vbox.addLayout(hbox2)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateBinaryImageParams(self):
        """检查选择那种求阈值方法"""
        self.binary_image_flag = self.binary_image_input_radiobtn.isChecked()

        if self.binary_image_flag:
            threshold = float(self.binary_image_threshold_line_edit.text())
            if threshold > 255:
                threshold = 255
            elif threshold < 0:
                threshold = 0
        else:
            data = normalizeToGrayScale(self.data)
            threshold = self.binary_image_threshold_methods[self.binary_image_method_combx.currentText()](data)

        self.binary_image_threshold_method_index = self.binary_image_method_combx.currentIndex()
        self.binary_image_threshold = threshold

    # """------------------------------------------------------------------------------------------------------------"""
    """计算数据特征调用的函数"""

    def plotTimeDomainFeature(self):
        """绘制选中的时域特征图像"""
        features = calculateTimeDomainFeatures(self.data)
        self.plotFeature(features)

    def plotFrequencyDomainFeature(self):
        """绘制选中的频域特征图像"""
        features = calculateFrequencyDomainFeatures(self.data, self.sampling_rate)
        self.plotFeature(features)

    def plotFeature(self, features: Dict) -> None:
        """获取要计算的数据特征名字和值"""
        feature_name = self.plot_menu.sender().text()
        feature = features[feature_name]

        plot_widget = MyPlotWidget(feature_name + '图', '通道', '')
        x = self.xAxis(begin=1, end=self.current_channels, num=self.current_channels)
        x *= self.sampling_rate
        plot_widget.draw(x, feature, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'{feature_name}图')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制多通道云图调用函数"""

    def plotMultiWavesImage(self):
        """绘制多通道云图"""
        plot_widget = MyPlotWidget('多通道云图', '时间（s）', '通道', check_mouse=False)
        x = self.xAxis()
        colors = cycle(['red', 'lime', 'deepskyblue', 'yellow', 'plum', 'gold', 'blue', 'fuchsia', 'aqua', 'orange'])
        for i in range(1, self.current_channels + 1):
            plot_widget.draw(x, self.data[i - 1] + i, pen=QColor(next(colors)))  # 根据通道数个位选择颜色绘图
        self.tab_widget.addTab(plot_widget, '多通道云图')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制应变图调用函数"""

    def plotStrain(self):
        """将相位差转为应变率再积分"""
        data = self.data[self.channel_number - 1]
        x = self.xAxis()
        data = integrate.cumtrapz(data, x, initial=0) * 1e6
        plot_widget = MyPlotWidget('应变图', '时间（s）', '应变（με）', grid=True)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'应变图 - 通道号={self.channel_number}')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制PSD调用函数"""

    def plotPSD(self):
        """绘制psd图线"""
        data = self.data[self.channel_number - 1]
        data = self.window_method(self.current_sampling_times) * data
        mag = np.abs(np.fft.fft(data)[:self.current_sampling_times // 2])
        ps = mag ** 2 / self.current_sampling_times
        data = 20.0 * np.log10(ps / self.sampling_rate + 1e-9)  # 转dB单位
        plot_widget = MyPlotWidget('功率谱密度图', '频率（Hz）', '功率/频率（dB/Hz）', grid=True)
        x = self.xAxis(freq=True)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'功率谱密度图 - 窗口类型={self.window_text}\t'
                                            f'通道号={self.channel_number}')

    def plot2dPSD(self):
        """绘制2dpsd谱"""
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'功率谱密度 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')
        data = self.data[self.channel_number - 1]
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='density', mode='psd')
        plt.pcolormesh(t, f, 20.0 * np.log10(Sxx + 1e-5), cmap='viridis')
        plt.colorbar(label='功率/频率（dB/Hz）')
        plt.title('功率谱密度')
        plt.xlabel('时间（s）')
        plt.ylabel('频率（Hz）')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dPSD(self):
        """绘制3dpsd"""
        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'三维功率谱密度 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='density', mode='psd')
        im = ax.plot_surface(f[:, None], t[None, :], 20.0 * np.log10(Sxx + 1e-5), cmap='viridis')
        plt.colorbar(im, ax=ax, label='功率/频率（dB/Hz）', pad=0.2)
        ax.set_title('三维功率谱密度')
        ax.set_xlabel('频率（Hz）')
        ax.set_ylabel('时间（s）')
        ax.set_zlabel('功率/频率（dB/Hz）')
        plt.xlim(0, self.sampling_rate / 2)

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制各种谱调用函数"""

    def plotMagnitudeSpectrum(self):
        """绘制幅度谱"""
        data = self.data[self.channel_number - 1]
        data = self.window_method(self.current_sampling_times) * data
        data = 20.0 * np.log10(
            np.abs(np.fft.fft(data)[:self.current_sampling_times // 2]) + 1e-5)
        plot_widget = MyPlotWidget('幅度谱', '频率（Hz）', '幅度（dB）', grid=True)
        x = self.xAxis(freq=True)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'幅度谱 - 窗口类型={self.window_text}\t'
                                            f'通道号={self.channel_number}')

    def plot2dMagnitudeSpectrum(self):
        """绘制2d幅度谱"""
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'幅度谱 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')

        data = self.data[self.channel_number - 1]
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='magnitude')
        plt.pcolormesh(t, f, 20.0 * np.log10(Sxx + 1e-5), cmap='viridis')
        plt.colorbar(label='幅度（dB）')
        plt.title('幅度谱')
        plt.xlabel('时间（s）')
        plt.ylabel('频率（Hz）')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dMagnitudeSpectrum(self):
        """绘制3d幅度谱"""
        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'三维幅度谱 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='magnitude')
        im = ax.plot_surface(f[:, None], t[None, :], 20.0 * np.log10(Sxx + 1e-5), cmap='viridis')
        plt.colorbar(im, ax=ax, label='幅度（dB）', pad=0.2)
        ax.set_title('三维幅度谱')
        ax.set_xlabel('频率（Hz）')
        ax.set_ylabel('时间（s）')
        ax.set_zlabel('幅度（dB）')
        plt.xlim(0, self.sampling_rate / 2)

    def plotAngleSpectrum(self):
        """绘制角度谱"""
        data = self.data[self.channel_number - 1]
        data = self.window_method(self.current_sampling_times) * data
        data = np.angle(np.fft.fft(data)[:self.current_sampling_times // 2])
        plot_widget = MyPlotWidget('角度谱', '频率（Hz）', '角度（rad）', grid=True)
        x = self.xAxis(freq=True)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'角度谱 - 窗口类型={self.window_text}\t'
                                            f'通道号={self.channel_number}')

    def plot2dAngleSpectrum(self):
        """绘制2d角度谱"""
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'角度谱 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')

        data = self.data[self.channel_number - 1]
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='angle')
        plt.pcolormesh(t, f, Sxx, cmap='viridis')
        plt.colorbar(label='角度（rad）')
        plt.title('角度谱')
        plt.xlabel('时间（s）')
        plt.ylabel('频率（Hz）')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dAngleSpectrum(self):
        """绘制3d角度谱"""
        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        figure_widget = FigureCanvas(figure)
        self.tab_widget.addTab(figure_widget, f'三维角度谱 - 窗口类型={self.window_text}\t'
                                              f'通道号={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='angle')
        im = ax.plot_surface(f[:, None], t[None, :], Sxx, cmap='viridis')
        plt.colorbar(im, ax=ax, label='角度（rad）', pad=0.2)
        ax.set_title('三维角度谱')
        ax.set_xlabel('频率（Hz）')
        ax.set_ylabel('时间（s）')
        ax.set_zlabel('角度（rad）')
        plt.xlim(0, self.sampling_rate / 2)

    # """------------------------------------------------------------------------------------------------------------"""
    """操作-加窗设置调用函数"""

    def windowOptionsDialog(self):
        """加窗设置调用窗口"""
        dialog = Dialog()
        dialog.setWindowTitle('加窗设置')

        window_method_label = Label('窗口类型')
        self.window_method_combx = ComboBox()
        self.window_method_combx.addItems(self.window_methods.keys())
        self.window_method_combx.setCurrentText(self.window_text)

        window_length_label = Label('窗长')
        self.window_length_line_edit = LineEditWithReg()
        self.window_length_line_edit.setText(str(self.window_length))
        self.window_length_line_edit.setToolTip('通常为2的倍数')

        window_overlap_size_ratio_label = Label('窗口重叠比例')
        self.window_overlap_size_ratio_line_edit = LineEditWithReg(digit=True)
        self.window_overlap_size_ratio_line_edit.setText(str(self.window_overlap_size_ratio))
        self.window_overlap_size_ratio_line_edit.setToolTip('相邻窗口的重叠比例，介于[0, 1)')

        btn = PushButton('确定')
        btn.clicked.connect(self.updateWindowOptionsParams)
        btn.clicked.connect(self.plotSingleChannelTime)
        btn.clicked.connect(self.plotAmplitudeFrequency)
        btn.clicked.connect(dialog.close)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox1.addWidget(window_method_label)
        hbox1.addStretch(1)
        hbox1.addWidget(self.window_method_combx)
        hbox2.addWidget(window_length_label)
        hbox2.addStretch(1)
        hbox2.addWidget(self.window_length_line_edit)
        hbox3.addWidget(window_overlap_size_ratio_label)
        hbox3.addStretch(1)
        hbox3.addWidget(self.window_overlap_size_ratio_line_edit)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateWindowOptionsParams(self):
        """更新加窗设置"""
        self.window_text = self.window_method_combx.currentText()
        self.window_method = self.window_methods[self.window_text]

        self.window_length = int(self.window_length_line_edit.text())

        if float(self.window_overlap_size_ratio_line_edit.text()) < 0:
            self.window_overlap_size_ratio = 0
        elif float(self.window_overlap_size_ratio_line_edit.text()) >= 1:
            self.window_overlap_size_ratio = 0.9999
        else:
            self.window_overlap_size_ratio = float(self.window_overlap_size_ratio_line_edit.text())
        self.window_overlap_size = int(round(self.window_overlap_size_ratio * self.window_length))

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter更新数据调用函数"""

    def updateFilteredData(self):
        """变更更新数据菜单"""
        if self.update_data:
            self.update_data_action.setText('更新数据（否）')
            self.update_data = False
        else:
            self.update_data_action.setText('更新数据（是）')
            self.update_data = True

    def ifUpdateData(self, flag: bool, data: np.array) -> None:
        """判断是否在滤波之后更新数据"""
        if flag:
            self.origin_data[self.channel_number - 1] = data
            self.data = self.origin_data
            self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""
    """滤波-EMD调用函数"""

    def plotEMD(self):
        """绘制emd分解图和重构图"""
        self.emd_method = self.emd_menu.sender().text()
        data = self.data[self.channel_number - 1]

        try:
            if self.emd_method == 'EMD':
                emd = EMD()
                self.imfs_res = emd.emd(data, max_imf=self.imfs_res_num - 1)
            elif self.emd_method == 'EEMD':
                emd = EEMD(trials=self.eemd_trials, noise_width=self.eemd_noise_width)
                self.imfs_res = emd.eemd(data, max_imf=self.imfs_res_num - 1)
            elif self.emd_method == 'CEEMDAN':
                if not hasattr(self, 'ceemdan_noise_kind_combx'):
                    noise_kind = 'normal'
                else:
                    noise_kind = self.ceemdan_noise_kind_combx.currentText()
                emd = CEEMDAN(trials=self.ceemdan_trials, epsilon=self.ceemdan_epsilon,
                              noise_scale=self.ceemdan_noise_scale,
                              noise_kind=noise_kind, range_thr=self.ceemdan_range_thr,
                              total_power_thr=self.ceemdan_total_power_thr)
                self.imfs_res = emd.ceemdan(data, max_imf=self.imfs_res_num - 1)
        except Exception as err:
            printError(err)

        if self.emd_options_flag:
            wgt = QWidget()
            wgt.setFixedWidth(self.tab_widget.width())
            vbox1 = QVBoxLayout()
            vbox2 = QVBoxLayout()
            hbox = QHBoxLayout()
            scroll_area = QScrollArea()

            pw_time_list, pw_fre_list = [], []
            for i in range(len(self.imfs_res)):
                if i == 0:
                    pw_time = MyPlotWidget(f'{self.emd_method}分解 - 时域', '', 'IMF1（rad）')
                    pw_fre = MyPlotWidget(f'{self.emd_method}分解 - 频域', '', 'IMF1')
                elif i == len(self.imfs_res) - 1:
                    pw_time = MyPlotWidget('', '时间（s）', f'Residual（rad）')
                    pw_fre = MyPlotWidget('', '频率（Hz）', f'Residual')
                else:
                    pw_time = MyPlotWidget('', '', f'IMF{i + 1}（rad）')
                    pw_fre = MyPlotWidget('', '', f'IMF{i + 1}')

                x_time = self.xAxis()
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, self.imfs_res[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])  # 设置时域x轴对应

                data = toAmplitude(self.imfs_res[i])
                x_fre = self.xAxis(freq=True)
                pw_fre.setFixedHeight(150)
                pw_fre.draw(x_fre, data, pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])  # 设置频域x轴对应

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area, f'{self.emd_method} - 分解: IMF数量={self.imfs_res_num - 1}\t'
                                                f'通道号={self.channel_number}')
        else:
            reconstruct_imf = [int(i) for i in re.findall('\d+', self.reconstruct_nums)]  # 映射为整数类型
            data = np.zeros(self.imfs_res[0].shape)
            for i in range(len(reconstruct_imf) - 1):
                imf_num = reconstruct_imf[i]
                data += self.imfs_res[imf_num]  # 重构数据

            combine_widget = self.initTwoPlotWidgets(data, self.emd_method + '重构')

            self.tab_widget.addTab(combine_widget, f'{self.emd_method} - 重构: 重构IMF={reconstruct_imf}')

            self.ifUpdateData(self.update_data, data)

        self.emd_plot_ins_fre_action.setEnabled(True)

    def EMDOptionsDialog(self):
        """使用emd分解合成滤波"""
        dialog = Dialog()
        dialog.resize(600, 200)
        dialog.setWindowTitle('EMD设置')

        shared_options_label = Label('共享设置')
        shared_options_label.setAlignment(Qt.AlignHCenter)
        self.emd_decompose_radio_btn = RadioButton('分解')
        self.emd_decompose_radio_btn.setChecked(self.emd_options_flag)
        imf_num_label = Label('IMF数量')
        self.emd_decompose_line_edit = LineEditWithReg()
        self.emd_decompose_line_edit.setToolTip('IMF数量，最大为9')
        self.emd_decompose_line_edit.setText(str(self.imfs_res_num - 1))

        self.emd_reconstruct_radio_btn = RadioButton('重构')
        self.emd_reconstruct_radio_btn.setChecked(not self.emd_options_flag)
        reconstruct_imf_number_label = Label('重构的IMF')
        self.emd_reconstruct_line_edit = LineEdit()
        self.emd_reconstruct_line_edit.setToolTip('重构的IMF号，应用逗号或空格分隔')
        self.emd_reconstruct_line_edit.setText(str(self.reconstruct_nums))

        if not hasattr(self, 'imfs_res'):
            self.emd_reconstruct_radio_btn.setEnabled(False)
            reconstruct_imf_number_label.setEnabled(False)
            self.emd_reconstruct_line_edit.setEnabled(False)

        eemd_options_label = Label('EEMD设置')
        eemd_options_label.setAlignment(Qt.AlignHCenter)
        eemd_trials_label = Label('试验点')
        self.eemd_trials_line_edit = LineEditWithReg()
        self.eemd_trials_line_edit.setText(str(self.eemd_trials))
        self.eemd_trials_line_edit.setToolTip('添加噪声的试验点或施加EMD点的数量')
        eemd_noise_width_label = Label('噪声宽度')
        self.eemd_noise_width_line_edit = LineEditWithReg(digit=True)
        self.eemd_noise_width_line_edit.setText(str(self.eemd_noise_width))
        self.eemd_noise_width_line_edit.setToolTip('高斯噪声的标准差，与信号的幅值有关')

        ceemdan_options_label = Label('CEEMDAN设置')
        ceemdan_options_label.setAlignment(Qt.AlignHCenter)
        ceemdan_trials_label = Label('试验点')
        self.ceemdan_trials_line_edit = LineEditWithReg()
        self.ceemdan_trials_line_edit.setText(str(self.ceemdan_trials))
        self.ceemdan_trials_line_edit.setToolTip('添加噪声的试验点或施加EMD点的数量')
        ceemdan_epsilon_label = Label('Epsilon')
        self.ceemdan_epsilon_line_edit = LineEditWithReg(digit=True)
        self.ceemdan_epsilon_line_edit.setText(str(self.ceemdan_epsilon))
        self.ceemdan_epsilon_line_edit.setToolTip('添加噪声乘标准差后的大小')
        ceemdan_noise_scale_label = Label('噪声大小')
        self.ceemdan_noise_scale_line_edit = LineEditWithReg(digit=True)
        self.ceemdan_noise_scale_line_edit.setText(str(self.ceemdan_noise_scale))
        self.ceemdan_noise_scale_line_edit.setToolTip('添加噪声的振幅')
        ceemdan_noise_kind_label = Label('噪声种类')
        self.ceemdan_noise_kind_combx = ComboBox()
        self.ceemdan_noise_kind_combx.addItems(['normal', 'uniform'])
        self.ceemdan_noise_kind_combx.setCurrentIndex(self.ceemdan_noise_kind_index)
        ceemdan_range_thr_label = Label('振幅范围阈值')
        self.ceemdan_range_thr_line_edit = LineEditWithReg(digit=True)
        self.ceemdan_range_thr_line_edit.setText(str(self.ceemdan_range_thr))
        self.ceemdan_range_thr_line_edit.setToolTip('用于IMF分解检查，其值等于与初始信号振幅之比的百分数，如果绝对振幅小于振幅范围阈值，'
                                                    '则认为分解完成')
        ceemdan_total_power_thr_label = Label('总功率阈值')
        self.ceemdan_total_power_thr_line_edit = LineEditWithReg(digit=True)
        self.ceemdan_total_power_thr_line_edit.setText(str(self.ceemdan_total_power_thr))
        self.ceemdan_total_power_thr_line_edit.setToolTip('用于IMF分解检查，如果信号总功率小于总功率阈值，则认为分解完成')

        btn = PushButton('确定')
        btn.clicked.connect(self.updateEMDParams)
        btn.clicked.connect(dialog.close)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        shared_options_vbox = QVBoxLayout()

        hbox3 = QHBoxLayout()
        eemd_options_vbox = QVBoxLayout()

        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()
        hbox6 = QHBoxLayout()
        ceemdan_options_vbox = QVBoxLayout()

        vbox = QVBoxLayout()

        hbox1.addWidget(self.emd_decompose_radio_btn)
        hbox1.addStretch(1)
        hbox1.addWidget(imf_num_label)
        hbox1.addWidget(self.emd_decompose_line_edit)
        hbox2.addWidget(self.emd_reconstruct_radio_btn)
        hbox2.addStretch(1)
        hbox2.addWidget(reconstruct_imf_number_label)
        hbox2.addWidget(self.emd_reconstruct_line_edit)
        shared_options_vbox.addWidget(shared_options_label)
        shared_options_vbox.addSpacing(10)
        shared_options_vbox.addLayout(hbox1)
        shared_options_vbox.addLayout(hbox2)

        hbox3.addWidget(eemd_trials_label)
        hbox3.addWidget(self.eemd_trials_line_edit)
        hbox3.addStretch(1)
        hbox3.addWidget(eemd_noise_width_label)
        hbox3.addWidget(self.eemd_noise_width_line_edit)
        eemd_options_vbox.addWidget(eemd_options_label)
        eemd_options_vbox.addSpacing(10)
        eemd_options_vbox.addLayout(hbox3)

        hbox4.addWidget(ceemdan_trials_label)
        hbox4.addWidget(self.ceemdan_trials_line_edit)
        hbox4.addStretch(1)
        hbox4.addWidget(ceemdan_epsilon_label)
        hbox4.addWidget(self.ceemdan_epsilon_line_edit)
        hbox5.addWidget(ceemdan_noise_scale_label)
        hbox5.addWidget(self.ceemdan_noise_scale_line_edit)
        hbox5.addStretch(1)
        hbox5.addWidget(ceemdan_noise_kind_label)
        hbox5.addWidget(self.ceemdan_noise_kind_combx)
        hbox6.addWidget(ceemdan_range_thr_label)
        hbox6.addWidget(self.ceemdan_range_thr_line_edit)
        hbox6.addStretch(1)
        hbox6.addWidget(ceemdan_total_power_thr_label)
        hbox6.addWidget(self.ceemdan_total_power_thr_line_edit)
        ceemdan_options_vbox.addWidget(ceemdan_options_label)
        ceemdan_options_vbox.addSpacing(10)
        ceemdan_options_vbox.addLayout(hbox4)
        ceemdan_options_vbox.addLayout(hbox5)
        ceemdan_options_vbox.addLayout(hbox6)

        vbox.addLayout(shared_options_vbox)
        vbox.addSpacing(30)
        vbox.addLayout(eemd_options_vbox)
        vbox.addSpacing(30)
        vbox.addLayout(ceemdan_options_vbox)
        vbox.addSpacing(20)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateEMDParams(self):
        """更新分解数和重构数"""
        self.emd_options_flag = self.emd_decompose_radio_btn.isChecked()

        if self.emd_options_flag:
            if int(self.emd_decompose_line_edit.text()) == 0:
                self.imfs_res_num = 2
            elif int(self.emd_decompose_line_edit.text()) >= 10:
                self.imfs_res_num = 10
            else:
                self.imfs_res_num = int(self.emd_decompose_line_edit.text()) + 1
        else:
            self.reconstruct_nums = ''.join(self.emd_reconstruct_line_edit.text())

        self.eemd_trials = int(self.eemd_trials_line_edit.text())
        self.eemd_noise_width = float(self.eemd_noise_width_line_edit.text())

        self.ceemdan_trials = int(self.ceemdan_trials_line_edit.text())
        self.ceemdan_epsilon = float(self.ceemdan_epsilon_line_edit.text())
        self.ceemdan_noise_scale = float(self.ceemdan_noise_scale_line_edit.text())
        self.ceemdan_noise_kind_index = self.ceemdan_noise_kind_combx.currentIndex()
        self.ceemdan_range_thr = float(self.ceemdan_range_thr_line_edit.text())
        self.ceemdan_total_power_thr = float(self.ceemdan_total_power_thr_line_edit.text())

    def plotEMDInstantaneousFrequency(self):
        """绘制瞬时频率"""
        x = self.xAxis()
        analytic_signal = hilbert(self.imfs_res)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        inst_freqs = np.diff(inst_phase) / (2 * np.pi * (x[1] - x[0]))
        inst_freqs = np.concatenate((inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)), axis=1)

        wgt = QWidget()
        wgt.setFixedWidth(self.tab_widget.width())
        vbox = QVBoxLayout()
        scroll_area = QScrollArea()

        pw_list = []
        for i in range(len(inst_freqs)):
            if i == 0:
                pw_list.append(MyPlotWidget('瞬时频率图', '', 'IMF1（Hz）'))
            elif i == len(inst_freqs) - 1:
                pw_list.append(MyPlotWidget('', '时间（s）', f'Residual'))
            else:
                pw_list.append(MyPlotWidget('', '', f'IMF{i + 1}（Hz）'))

            pw_list[i].setXLink(pw_list[0])
            pw_list[i].draw(x, inst_freqs[i], pen=QColor('blue'))
            pw_list[i].setFixedHeight(150)
            vbox.addWidget(pw_list[i])
        wgt.setLayout(vbox)
        scroll_area.setWidget(wgt)
        self.tab_widget.addTab(scroll_area, f'{self.emd_method} - 瞬时频率:\t'
                                            f'通道号={self.channel_number}')

    # """------------------------------------------------------------------------------------------------------------"""
    """滤波-IIR滤波器-Butterworth调用函数"""

    def iirFilterDesign(self):
        name = self.filter_menu.sender().text()
        if not self.filter or self.filter.filter_name != name:
            self.filter = Filter(self, name)
        self.filter.runDialog()

    # """------------------------------------------------------------------------------------------------------------"""
    """小波菜单调用函数"""

    def waveletDWTDialog(self):
        """设置选择的小波、分解层数、填充模式"""
        dialog = Dialog()
        dialog.setWindowTitle('离散小波变换')

        self.wavelet_dwt_decompose_radiobtn = RadioButton('分解')
        self.wavelet_dwt_decompose_radiobtn.setChecked(self.wavelet_dwt_flag)

        wavelet_dwt_reconstruct_radiobtn = RadioButton('重构')
        wavelet_dwt_reconstruct_radiobtn.setChecked(not self.wavelet_dwt_flag)

        wavelet_dwt_reconstruct_label = Label('重构系数')
        self.wavelet_dwt_reconstruct_line_edit = LineEdit()
        self.wavelet_dwt_reconstruct_line_edit.setFixedWidth(500)
        self.wavelet_dwt_reconstruct_line_edit.setToolTip('选择cAn和cDn系数进行重构，cAn为近似系数，cDn-cD1为细节系数')
        self.wavelet_dwt_reconstruct_line_edit.setText(str(self.wavelet_dwt_reconstruct))

        if not hasattr(self, 'wavelet_dwt_coeffs'):
            wavelet_dwt_reconstruct_radiobtn.setEnabled(False)
            wavelet_dwt_reconstruct_label.setEnabled(False)
            self.wavelet_dwt_reconstruct_line_edit.setEnabled(False)

        wavelet_dwt_label = Label('选择小波：')

        wavelet_dwt_family_label = Label('小波族')
        self.wavelet_dwt_family_combx = ComboBox()
        self.wavelet_dwt_family_combx.addItems(pywt.families(short=False))
        self.wavelet_dwt_family_combx.setCurrentIndex(self.wavelet_dwt_family_index)
        self.wavelet_dwt_family_combx.currentIndexChanged[int].connect(self.waveletDWTChangeNameComboBox)

        wavelet_dwt_name_label = Label('小波名称')
        self.wavelet_dwt_name_combx = ComboBox()
        self.wavelet_dwt_name_combx.setFixedWidth(75)
        self.wavelet_dwt_name_combx.addItems(pywt.wavelist(pywt.families()[self.wavelet_dwt_family_index]))
        self.wavelet_dwt_name_combx.setCurrentIndex(self.wavelet_dwt_name_index)

        wavelet_dwt_decompose_level_label = Label('分解层数')
        self.wavelet_dwt_decompose_level_line_edit = LineEditWithReg()
        self.wavelet_dwt_decompose_level_line_edit.setToolTip('分解层数')
        self.wavelet_dwt_decompose_level_line_edit.setText(str(self.wavelet_dwt_decompose_level))

        self.wavelet_dwt_decompose_level_checkbx = CheckBox('使用计算的分解层数')
        self.wavelet_dwt_decompose_level_checkbx.setToolTip('使用根据数据长度和选择的小波计算得到的分解层数')
        self.wavelet_dwt_decompose_level_checkbx.setChecked(self.wavelet_dwt_decompose_level_calculated)

        wavelet_dwt_padding_mode_label = Label('填充模式')
        self.wavelet_dwt_padding_mode_combx = ComboBox()
        self.wavelet_dwt_padding_mode_combx.setToolTip('数据延长模式')
        self.wavelet_dwt_padding_mode_combx.addItems(pywt.Modes.modes)
        self.wavelet_dwt_padding_mode_combx.setCurrentIndex(self.wavelet_dwt_padding_mode_index)

        btn = PushButton('确定')
        btn.clicked.connect(self.updateWaveletDWTParams)
        btn.clicked.connect(self.runWaveletDWT)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()

        hbox1.addWidget(self.wavelet_dwt_decompose_radiobtn)
        hbox1.addStretch(1)
        hbox1.addWidget(wavelet_dwt_label)
        hbox1.addWidget(wavelet_dwt_family_label)
        hbox1.addWidget(self.wavelet_dwt_family_combx)
        hbox1.addWidget(wavelet_dwt_name_label)
        hbox1.addWidget(self.wavelet_dwt_name_combx)

        hbox2.addWidget(wavelet_dwt_reconstruct_radiobtn)
        hbox2.addStretch(1)
        hbox2.addWidget(wavelet_dwt_reconstruct_label)
        hbox2.addWidget(self.wavelet_dwt_reconstruct_line_edit)

        hbox3.addWidget(wavelet_dwt_decompose_level_label)
        hbox3.addWidget(self.wavelet_dwt_decompose_level_line_edit)
        hbox3.addWidget(self.wavelet_dwt_decompose_level_checkbx)
        hbox3.addStretch(1)
        hbox3.addWidget(wavelet_dwt_padding_mode_label)
        hbox3.addWidget(self.wavelet_dwt_padding_mode_combx)

        vbox.addSpacing(10)
        vbox.addLayout(hbox1)
        vbox.addSpacing(10)
        vbox.addLayout(hbox3)
        vbox.addSpacing(10)
        vbox.addLayout(hbox2)
        vbox.addSpacing(10)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def waveletDWTChangeNameComboBox(self, index: int) -> None:
        """根据选择的小波族更改小波的名字"""
        self.wavelet_dwt_name_index = 0
        self.wavelet_dwt_name_combx.clear()
        self.wavelet_dwt_name_combx.addItems(pywt.wavelist(pywt.families()[index]))
        self.wavelet_dwt_name_combx.setCurrentIndex(self.wavelet_dwt_name_index)

    def updateWaveletDWTParams(self):
        """更新参数"""
        self.wavelet_dwt_flag = self.wavelet_dwt_decompose_radiobtn.isChecked()
        self.wavelet_dwt_reconstruct = self.wavelet_dwt_reconstruct_line_edit.text()
        self.wavelet_dwt_family_index = self.wavelet_dwt_family_combx.currentIndex()
        self.wavelet_dwt_name_index = self.wavelet_dwt_name_combx.currentIndex()
        self.wavelet_dwt_decompose_level = int(self.wavelet_dwt_decompose_level_line_edit.text())
        self.wavelet_dwt_decompose_level_calculated = self.wavelet_dwt_decompose_level_checkbx.isChecked()
        self.wavelet_dwt_padding_mode_index = self.wavelet_dwt_padding_mode_combx.currentIndex()

    def runWaveletDWT(self):
        """分解或重构"""
        data = self.data[self.channel_number - 1]

        if self.wavelet_dwt_decompose_level_calculated:
            self.wavelet_dwt_decompose_level = pywt.dwt_max_level(self.current_sampling_times,
                                                                  self.wavelet_dwt_name_combx.currentText())  # 求最大分解层数

        if self.wavelet_dwt_flag:
            try:
                self.wavelet_dwt_coeffs = pywt.wavedec(data, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                                       mode=self.wavelet_dwt_padding_mode_combx.currentText(),
                                                       level=self.wavelet_dwt_decompose_level)  # 求分解系数
                self.wavelet_dwt_reconstruct = []
                self.wavelet_dwt_reconstruct.append(f'cA{self.wavelet_dwt_decompose_level}')
                for i in range(len(self.wavelet_dwt_coeffs) - 1, 0, -1):
                    self.wavelet_dwt_reconstruct.append(f'cD{i}')
                self.wavelet_dwt_former_reconstruct = self.wavelet_dwt_reconstruct
            except Exception as err:
                printError(err)

        else:
            rec_coeffs_split = str(self.wavelet_dwt_reconstruct).split("'")
            rec_coeffs = []
            for i in rec_coeffs_split:
                coeff = re.match('^\w{2}\d+$', i)
                if coeff is not None:
                    rec_coeffs.append(coeff.group())
            self.wavelet_dwt_reconstruct = rec_coeffs  # 更新规范的重构系数显示

            for i in self.wavelet_dwt_former_reconstruct:
                if i not in rec_coeffs:  # 删除的系数置0
                    if i == f'cA{self.wavelet_dwt_decompose_level}':
                        self.wavelet_dwt_coeffs[0] = np.zeros_like(self.wavelet_dwt_coeffs[0])
                    else:
                        number = int(re.match('^cD(\d+)$', i).group(1))
                        self.wavelet_dwt_coeffs[-number] = np.zeros_like(self.wavelet_dwt_coeffs[-number])

        self.plotWaveletDWT(self.wavelet_dwt_coeffs)

    def plotWaveletDWT(self, coeffs: np.array) -> None:
        """绘图"""
        if self.wavelet_dwt_flag:
            wgt = QWidget()
            wgt.setFixedWidth(self.tab_widget.width())
            vbox1 = QVBoxLayout()
            vbox2 = QVBoxLayout()
            hbox = QHBoxLayout()
            scroll_area = QScrollArea()

            pw_time_list, pw_fre_list = [], []
            for i in range(len(coeffs)):
                if i == 0:
                    pw_time = MyPlotWidget('离散小波变换分解 - 时域', '', f'cA{self.wavelet_dwt_decompose_level}（rad）')
                    pw_fre = MyPlotWidget('离散小波变换分解 - 频域', '', f'cA{self.wavelet_dwt_decompose_level}')
                elif i == len(coeffs) - 1:
                    pw_time = MyPlotWidget('', '时间（s）', f'cD1（rad）')
                    pw_fre = MyPlotWidget('', '频率（Hz）', f'cD1')
                else:
                    pw_time = MyPlotWidget('', '', f'cD{len(coeffs) - i}（rad）')
                    pw_fre = MyPlotWidget('', '', f'cD{len(coeffs) - i}')

                x_time = self.xAxis(end=self.sampling_times_from_num + len(coeffs[i]), num=len(coeffs[i]))
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, coeffs[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])

                data = toAmplitude(coeffs[i])
                x_fre = self.xAxis(num=len(coeffs[i]), freq=True)
                pw_fre.setFixedHeight(150)
                pw_fre.draw(x_fre, data, pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area, f'离散小波变换 - 分解: 分解层数={self.wavelet_dwt_decompose_level}\t'
                                                f'小波={self.wavelet_dwt_name_combx.currentText()}\t'
                                                f'通道号={self.channel_number}')
        else:
            try:
                data = pywt.waverec(coeffs, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                    mode=self.wavelet_dwt_padding_mode_combx.currentText())  # 重构信号
                combine_widget = self.initTwoPlotWidgets(data, '离散小波变换重构')

                self.tab_widget.addTab(combine_widget, f'离散小波变换 -重构: 系数={self.wavelet_dwt_reconstruct}\t'
                                                       f'小波={self.wavelet_dwt_name_combx.currentText()}\t'
                                                       f'通道号={self.channel_number}')

                self.ifUpdateData(self.update_data, data)
            except Exception as err:
                printError(err)

    def waveletThresholdDialog(self):
        """小波去噪"""
        dialog = Dialog()
        dialog.setWindowTitle('小波去噪')

        wavelet_threshold_label = Label('阈值')
        self.wavelet_threshold_line_edit = LineEditWithReg(digit=True)
        self.wavelet_threshold_line_edit.setToolTip('去噪阈值')
        self.wavelet_threshold_line_edit.setText(str(self.wavelet_threshold))

        wavelet_threshold_sub_label = Label('替换值')
        self.wavelet_threshold_sub_line_edit = LineEditWithReg(digit=True)
        self.wavelet_threshold_sub_line_edit.setToolTip('数据中筛除的值替换为该值')
        self.wavelet_threshold_sub_line_edit.setText(str(self.wavelet_threshold_sub))

        wavelet_threshold_mode_label = Label('阈值类型')
        self.wavelet_threshold_mode_combx = ComboBox()
        self.wavelet_threshold_mode_combx.setToolTip('设置阈值类型')
        self.wavelet_threshold_mode_combx.addItems(self.wavelet_threshold_modes)
        self.wavelet_threshold_mode_combx.setCurrentIndex(self.wavelet_threshold_mode_index)

        btn = PushButton('确定')
        btn.clicked.connect(self.updateWaveletThresholdParams)
        btn.clicked.connect(self.plotWaveletThreshold)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(wavelet_threshold_label)
        hbox.addWidget(self.wavelet_threshold_line_edit)
        hbox.addWidget(wavelet_threshold_sub_label)
        hbox.addWidget(self.wavelet_threshold_sub_line_edit)
        hbox.addStretch(1)
        hbox.addWidget(wavelet_threshold_mode_label)
        hbox.addWidget(self.wavelet_threshold_mode_combx)

        vbox.addSpacing(5)
        vbox.addLayout(hbox)
        vbox.addSpacing(5)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def updateWaveletThresholdParams(self):
        """更新小波阈值"""
        self.wavelet_threshold = float(self.wavelet_threshold_line_edit.text())
        self.wavelet_threshold_sub = float(self.wavelet_threshold_sub_line_edit.text())
        self.wavelet_threshold_mode_index = self.wavelet_threshold_mode_combx.currentIndex()

    def plotWaveletThreshold(self):
        """绘制滤波后的图像"""
        data = self.data[self.channel_number - 1]

        try:
            data = pywt.threshold(data, value=self.wavelet_threshold,
                                  mode=self.wavelet_threshold_mode_combx.currentText(),
                                  substitute=self.wavelet_threshold_sub)  # 阈值滤波
            combine_widget = self.initTwoPlotWidgets(data, '小波去噪')

            self.tab_widget.addTab(combine_widget, f'小波去噪 - 阈值={self.wavelet_threshold}\t'
                                                   f'阈值类型={self.wavelet_threshold_mode_combx.currentText()}\t'
                                                   f'通道号={self.channel_number}')

            self.ifUpdateData(self.update_data, data)
        except Exception as err:
            printError(err)

    def waveletPacketsDialog(self):
        """小波包分解"""
        dialog = Dialog()
        dialog.setWindowTitle('小波包')

        self.wavelet_packets_decompose_radiobtn = RadioButton('分解')
        self.wavelet_packets_decompose_radiobtn.setChecked(self.wavelet_packets_flag)

        wavelet_packets_reconstruct_radiobtn = RadioButton('重构')
        wavelet_packets_reconstruct_radiobtn.setChecked(not self.wavelet_packets_flag)

        wavelet_packets_reconstruct_label = Label('重构子节点')
        self.wavelet_packets_reconstruct_line_edit = LineEdit()
        self.wavelet_packets_reconstruct_line_edit.setFixedWidth(500)
        self.wavelet_packets_reconstruct_line_edit.setToolTip('选择重构的子节点，a为近似节点路径分支，d为细节节点路径分支，'
                                                              '多个a或d相连代表了子节点路径')

        self.wavelet_packets_reconstruct_line_edit.setText(str(self.wavelet_packets_reconstruct))

        if not hasattr(self, 'wavelet_packets_subnodes'):
            wavelet_packets_reconstruct_radiobtn.setEnabled(False)
            wavelet_packets_reconstruct_label.setEnabled(False)
            self.wavelet_packets_reconstruct_line_edit.setEnabled(False)

        wavelet_packets_label = Label('选择小波：')

        wavelet_packets_family_label = Label('小波族')
        self.wavelet_packets_family_combx = ComboBox()
        self.wavelet_packets_family_combx.addItems(pywt.families(short=False))
        self.wavelet_packets_family_combx.setCurrentIndex(self.wavelet_packets_family_index)
        self.wavelet_packets_family_combx.currentIndexChanged[int].connect(self.waveletPacketsChangeNameComboBox)

        wavelet_packets_name_label = Label('小波名称')
        self.wavelet_packets_name_combx = ComboBox()
        self.wavelet_packets_name_combx.setFixedWidth(75)
        self.wavelet_packets_name_combx.addItems(pywt.wavelist(pywt.families()[self.wavelet_packets_family_index]))
        self.wavelet_packets_name_combx.setCurrentIndex(self.wavelet_packets_name_index)
        self.wavelet_packets_name_combx.currentIndexChanged.connect(self.waveletPacketsCalculateDecomposeMaxLevel)

        wavelet_packets_decompose_level_label = Label('分解层数')
        self.wavelet_packets_decompose_level_line_edit = LineEditWithReg()
        self.wavelet_packets_decompose_level_line_edit.setToolTip('分解层数')
        self.wavelet_packets_decompose_level_line_edit.setText(str(self.wavelet_packets_decompose_level))

        wavelet_packets_decompose_max_level_label = Label('最大分解层数')
        self.wavelet_packets_decompose_max_level_line_edit = LineEditWithReg(focus=False)
        self.wavelet_packets_decompose_max_level_line_edit.setToolTip('数据的最大分解层数，与数据长度和选择的小波有关')
        self.wavelet_packets_decompose_max_level = pywt.dwt_max_level(self.data.shape[1],
                                                                      self.wavelet_packets_name_combx.currentText())
        self.wavelet_packets_decompose_max_level_line_edit.setText(str(self.wavelet_packets_decompose_max_level))

        wavelet_packets_padding_mode_label = Label('填充模式')
        self.wavelet_packets_padding_mode_combx = ComboBox()
        self.wavelet_packets_padding_mode_combx.setToolTip('信号延长模式')
        self.wavelet_packets_padding_mode_combx.addItems(pywt.Modes.modes)
        self.wavelet_packets_padding_mode_combx.setCurrentIndex(self.wavelet_packets_padding_mode_index)

        btn = PushButton('确定')
        btn.clicked.connect(self.updateWaveletPacketsParams)
        btn.clicked.connect(self.runWaveletPackets)
        btn.clicked.connect(dialog.close)

        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()

        hbox1.addWidget(self.wavelet_packets_decompose_radiobtn)
        hbox1.addStretch(1)
        hbox1.addWidget(wavelet_packets_label)
        hbox1.addWidget(wavelet_packets_family_label)
        hbox1.addWidget(self.wavelet_packets_family_combx)
        hbox1.addWidget(wavelet_packets_name_label)
        hbox1.addWidget(self.wavelet_packets_name_combx)

        hbox2.addWidget(wavelet_packets_reconstruct_radiobtn)
        hbox2.addStretch(1)
        hbox2.addWidget(wavelet_packets_reconstruct_label)
        hbox2.addWidget(self.wavelet_packets_reconstruct_line_edit)

        hbox3.addWidget(wavelet_packets_decompose_level_label)
        hbox3.addWidget(self.wavelet_packets_decompose_level_line_edit)
        hbox3.addWidget(wavelet_packets_decompose_max_level_label)
        hbox3.addWidget(self.wavelet_packets_decompose_max_level_line_edit)
        hbox3.addStretch(1)
        hbox3.addWidget(wavelet_packets_padding_mode_label)
        hbox3.addWidget(self.wavelet_packets_padding_mode_combx)

        vbox.addLayout(hbox1)
        vbox.addSpacing(10)
        vbox.addLayout(hbox3)
        vbox.addSpacing(10)
        vbox.addLayout(hbox2)
        vbox.addSpacing(10)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def waveletPacketsChangeNameComboBox(self, index: int) -> None:
        """根据选择的小波族更改小波的名字"""
        self.wavelet_packets_name_index = 0
        self.wavelet_packets_name_combx.clear()
        self.wavelet_packets_name_combx.addItems(pywt.wavelist(pywt.families()[index]))
        self.wavelet_packets_name_combx.setCurrentIndex(self.wavelet_packets_name_index)

    def waveletPacketsCalculateDecomposeMaxLevel(self):
        """根据选择的小波计算最大分解层数"""
        wavelet = self.wavelet_packets_name_combx.currentText()
        if wavelet != '':
            self.wavelet_packets_decompose_max_level = pywt.dwt_max_level(self.data.shape[1], wavelet)  # 最大分解层数
            self.wavelet_packets_decompose_max_level_line_edit.setText(str(self.wavelet_packets_decompose_max_level))

    def updateWaveletPacketsParams(self):
        """更新小波包分解系数"""
        self.wavelet_packets_flag = self.wavelet_packets_decompose_radiobtn.isChecked()
        self.wavelet_packets_reconstruct = self.wavelet_packets_reconstruct_line_edit.text()
        self.wavelet_packets_family_index = self.wavelet_packets_family_combx.currentIndex()
        self.wavelet_packets_name_index = self.wavelet_packets_name_combx.currentIndex()
        self.wavelet_packets_decompose_level = int(self.wavelet_packets_decompose_level_line_edit.text())
        self.wavelet_packets_decompose_max_level = int(self.wavelet_packets_decompose_max_level_line_edit.text())
        self.wavelet_packets_padding_mode_index = self.wavelet_packets_padding_mode_combx.currentIndex()

    def runWaveletPackets(self):
        """处理数据，绘图"""
        data = self.data[self.channel_number - 1]

        try:
            if self.wavelet_packets_flag:
                self.wavelet_packets_wp = pywt.WaveletPacket(data,
                                                             wavelet=self.wavelet_packets_name_combx.currentText(),
                                                             mode=self.wavelet_packets_padding_mode_combx.currentText())  # 创建一个小波包
                self.wavelet_packets_subnodes = self.wavelet_packets_wp.get_level(
                    level=self.wavelet_packets_decompose_level, order='natural', decompose=True)  # 获得当前分解层数下的各节点
                self.wavelet_packets_reconstruct = [i.path for i in self.wavelet_packets_subnodes]

            else:
                total_paths = [i.path for i in self.wavelet_packets_subnodes]
                self.wavelet_packets_reconstruct = str(self.wavelet_packets_reconstruct).split("','")
                self.wavelet_packets_reconstruct = re.findall('\w+', self.wavelet_packets_reconstruct[0])

                for i in total_paths:
                    if i not in self.wavelet_packets_reconstruct:
                        del self.wavelet_packets_wp[i]
            self.plotWaveletPackets(self.wavelet_packets_subnodes)
        except Exception as err:
            printError(err)

    def plotWaveletPackets(self, subnodes: np.array) -> None:
        """绘图"""
        if self.wavelet_packets_flag:
            wgt = QWidget()
            wgt.setFixedWidth(self.tab_widget.width())
            vbox1 = QVBoxLayout()
            vbox2 = QVBoxLayout()
            hbox = QHBoxLayout()
            scroll_area = QScrollArea()

            pw_time_list, pw_fre_list = [], []
            for i in range(len(subnodes)):
                if i == 0:
                    pw_time = MyPlotWidget('小波包分解 - 时域', '', f'{self.wavelet_packets_subnodes[i].path}（rad）')
                    pw_fre = MyPlotWidget('小波包分解 - 频域', '', f'{self.wavelet_packets_subnodes[i].path}')
                elif i == len(subnodes) - 1:
                    pw_time = MyPlotWidget('', '时间（s）', f'{self.wavelet_packets_subnodes[i].path}（rad）')
                    pw_fre = MyPlotWidget('', '频率（Hz）', f'{self.wavelet_packets_subnodes[i].path}')
                else:
                    pw_time = MyPlotWidget('', '', f'{self.wavelet_packets_subnodes[i].path}（rad）')
                    pw_fre = MyPlotWidget('', '', f'{self.wavelet_packets_subnodes[i].path}')

                x_time = self.xAxis(end=self.sampling_times_from_num + len(subnodes[i].data),
                                    num=len(subnodes[i].data))
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, subnodes[i].data, pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])

                data = toAmplitude(subnodes[i].data)
                x_fre = self.xAxis(num=len(subnodes[i].data), freq=True)
                pw_fre.setFixedHeight(150)
                pw_fre.draw(x_fre, data, pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area, f'小波包 - 分解: 分解层数={self.wavelet_packets_decompose_level}\t'
                                                f'小波={self.wavelet_packets_name_combx.currentText()}\t'
                                                f'通道号={self.channel_number}')
        else:
            try:
                data = self.wavelet_packets_wp.reconstruct()  # 重构信号
                combine_widget = self.initTwoPlotWidgets(data, '小波包重构')

                self.tab_widget.addTab(combine_widget, f'小波包 - 重构: 子节点={self.wavelet_packets_reconstruct}\t'
                                                       f'小波={self.wavelet_packets_name_combx.currentText()}\t'
                                                       f'通道号={self.channel_number}')

                self.ifUpdateData(self.update_data, data)
            except Exception as err:
                printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """其他-筛选数据"""

    def dataSiftingDialog(self):
        if not self.data_sift:
            self.data_sift = DataSifting(self)
        self.data_sift.runDialog()

# """------------------------------------------------------------------------------------------------------------"""
