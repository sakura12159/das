"""
2023-8-14
ver1.4.3

1.修改数据范围
2.修复数据标签显示错误
3.修改搜索鼠标位置最近点的算法
4.修复绘制多通道云图视角问题
5.修改图片转py文件的方法
"""

import ctypes
import sys
import wave
from itertools import cycle

import pandas as pd
import pywt
from PyEMD import EMD, EEMD, CEEMDAN
from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, qApp, QTabWidget, QTableWidget, \
    QAbstractItemView, QTableWidgetItem, QHeaderView, QTabBar, QWidget, QScrollArea, QScrollBar
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import hilbert, filtfilt, spectrogram, detrend
from scipy.signal.windows import *

from filter import *
from function import *
from image import *
from widget import *


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initMainWindow()
        self.initGlobalParams()
        self.initUI()
        self.initMenu()
        self.initTabWidget()
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
        plt.rc('font', family='Times New Roman')
        # plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        # pg组件设置
        pg.setConfigOptions(leftButtonPan=True)  # 设置可用鼠标缩放
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')  # 设置界面前背景色

        # 输出设置
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)  # 设置输出时每行的长度

        self.channel_number = 1  # 当前通道
        self.channel_number_step = 1  # 通道号递增减步长
        self.files_read_number = 1  # 表格连续读取文件数

    def initUI(self):
        """初始化ui"""

        self.status_bar = self.statusBar()  # 状态栏
        self.status_bar.setStyleSheet('font-size: 15px; font-family: "Times New Roman";')
        self.menu_bar = self.menuBar()  # 菜单栏
        self.menu_bar.setStyleSheet('font-size: 18px; font-family: "Times New Roman";')
        self.setWindowTitle('Φ-DAS Visualizer')

        getPicture(icon_jpg, 'icon.jpg')
        self.setWindowIcon(QIcon('icon.jpg'))
        os.remove('icon.jpg')

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('myappid')  # 设置任务栏图标

    def initMenu(self):
        """初始化菜单"""

        # File
        self.file_menu = createMenu('File', self.menu_bar)

        # Import
        self.import_action = createAction('Import', self.file_menu, 'Import data file(s)', self.importData,
                                          short_cut='Ctrl+I')

        # File-Export
        self.export_action = createAction('Export', self.file_menu, 'Export data', self.exportData, short_cut='Ctrl+E')

        self.file_menu.addSeparator()

        # Quit
        self.quit_action = createAction('Quit', self.file_menu, 'Quit Φ-DAS Visualizer', qApp.quit, short_cut='Ctrl+Q')

        # Operation
        self.operation_menu = createMenu('Operation', self.menu_bar, enabled=False)

        # Operation-Calculate SNR
        self.calculate_snr_action = createAction('Calculate SNR', self.operation_menu,
                                                 'Calculate SNR of the chosen segment',
                                                 self.calculateSNRDialog)

        self.operation_menu.addSeparator()

        # Operation-Clip Data By Sampling Times
        self.clip_time_action = createAction('Clip Data By Sampling Times', self.operation_menu,
                                             'Clip data within a certain range of sampling times',
                                             self.clipSamplingTimesDialog)

        # Operation-Clip Data By Channels
        self.clip_channel_action = createAction('Clip Data By Channels', self.operation_menu,
                                                'Clip data within a certain range of channels', self.clipChannelsDialog)

        self.operation_menu.addSeparator()

        # Operation-Convert Data Unit To
        self.convert_menu = createMenu('Convert Data Unit To', self.operation_menu)

        # Operation-Convert Data Unit To-Phase Difference(rad)
        self.phase_difference_action = createAction('Phase Difference(rad)', self.convert_menu,
                                                    'Set phase difference as data unit', self.convertToPhaseDifference)
        self.phase_difference_action.setCheckable(True)
        self.phase_difference_action.setChecked(True)

        # Operation-Convert Data Unit To-Strain Rate(s^-1)
        self.strain_rate_action = createAction('Strain Rate(s^-1)', self.convert_menu, 'Set strain rate as data unit',
                                               self.convertToStrainRate)
        self.strain_rate_action.setCheckable(True)

        self.operation_menu.addSeparator()

        # Operation-Change Channel Number Step
        self.change_channel_number_step_action = createAction('Change Channel Number Step', self.operation_menu,
                                                              'Change the step when changing channel number',
                                                              self.changeChannelNumberStep)

        # Operation-Change Files Read Number
        self.change_files_read_number_action = createAction('Change Files Read Number', self.operation_menu,
                                                            'Change the number of files read when choosing file '
                                                            'from the table, counting from the chosen one',
                                                            self.changeFilesReadNumberDialog)

        # Plot
        self.plot_menu = createMenu('Plot', self.menu_bar, enabled=False)

        # Plot-Plot Binary Image
        self.plot_binary_image_action = createAction('Plot Binary Image', self.plot_menu,
                                                     'Set the threshold to plot binary image', self.binaryImageDialog)

        # Plot-Plot Data Features
        self.plot_data_features_menu = createMenu('Plot Data Features', self.plot_menu,
                                                  status_tip='Calculate and plot time-domain and '
                                                             'frequency-domain features of data')

        # Plot-Plot Data Features-Maximum Value etc.
        self.time_domain_chars_text = ['Maximum Value', 'Peak Value', 'Minimum Value', 'Mean Value',
                                       'Peak-To-Peak Value', 'Mean-Absolute Value', 'Root-Mean-Square',
                                       'Square-Root-Amplitude', 'Variance', 'Standard-Deviation', 'Kurtosis',
                                       'Skewness', 'Clearance Factor', 'Shape Factor', 'Impulse Factor', 'Crest Factor',
                                       'Kurtosis Factor']

        self.fre_domain_chars_text = ['Centroid Frequency', 'Mean Frequency', 'Root-Mean-Square Frequency',
                                      'Frequency Variance', 'Mean-Square Frequency', 'Frequency Standard-Deviation']

        for i in self.time_domain_chars_text:
            _ = createAction(i, self.plot_data_features_menu, f'Calculate {i.lower()}', self.plotTimeDomainFeature)

        # 时域和频域特征之间的分隔线
        self.plot_data_features_menu.addSeparator()

        for i in self.fre_domain_chars_text:
            _ = createAction(i, self.plot_data_features_menu, f'Calculate {i.lower()}',
                             self.plotFrequencyDomainFeature)

        # Plot-Plot Multi-Channel Image
        self.plot_multichannel_image_action = createAction('Plot Multi-Channel Image', self.plot_menu,
                                                           'Plot multi-channel image', self.plotMultiChannelImage)

        # Plot-Plot Strain Image
        self.plot_strain_image_action = createAction('Plot Strain Image', self.plot_menu,
                                                     'Plot strain image in microstrain', self.plotStrain)

        self.plot_menu.addSeparator()

        # Plot-Plot PSD
        self.plot_psd_menu = createMenu('Plot PSD', self.plot_menu)

        # Plot-Plot PSD-Plot PSD
        self.plot_psd_action = createAction('Plot PSD', self.plot_psd_menu, 'Plot psd', self.plotPSD)

        # Plot-Plot PSD-Plot 2D PSD
        self.plot_2d_psd_action = createAction('Plot 2D PSD', self.plot_psd_menu, 'Plot 2d psd', self.plot2dPSD)

        # Plot-Plot PSD-Plot 3D PSD
        self.plot_3d_psd_action = createAction('Plot 3D PSD', self.plot_psd_menu, 'Plot 3d psd', self.plot3dPSD)

        # Plot-Plot Spectrum
        self.plot_spectrum_menu = createMenu('Plot Spectrum', self.plot_menu)

        # Plot-Plot Spectrum-Plot Magnitude Spectrum
        self.plot_mag_spectrum_action = createAction('Plot Magnitude Spectrum', self.plot_spectrum_menu,
                                                     'Plot magnitude spectrum', self.plotMagnitudeSpectrum)

        # Plot-Plot Spectrum-Plot 2D Magnitude Spectrum
        self.plot_2d_mag_spectrum_action = createAction('Plot 2D Magnitude Spectrum', self.plot_spectrum_menu,
                                                        'Plot 2d magnitude spectrum', self.plot2dMagnitudeSpectrum)

        # Plot-Plot Spectrum-Plot 3D Magnitude Spectrum
        self.plot_3d_mag_psd_action = createAction('Plot 3D Magnitude Spectrum', self.plot_spectrum_menu,
                                                   'Plot 3d magnitude spectrum', self.plot3dMagnitudeSpectrum)

        self.plot_spectrum_menu.addSeparator()

        # Plot-Plot Spectrum-Plot Angle Spectrum
        self.plot_ang_spectrum_action = createAction('Plot Angle Spectrum', self.plot_spectrum_menu,
                                                     'Plot angle spectrum', self.plotAngleSpectrum)

        # Plot-Plot Spectrum-Plot 2D Angle Spectrum
        self.plot_2d_ang_spectrum_action = createAction('Plot 2D Angle Spectrum', self.plot_spectrum_menu,
                                                        'Plot 2d angle spectrum', self.plot2dAngleSpectrum)

        # Plot-Plot Spectrum-Plot 3D Angle Spectrum
        self.plot_3d_ang_spectrum_action = createAction('Plot 3D Angle Spectrum', self.plot_spectrum_menu,
                                                        'Plot 3d angle spectrum', self.plot3dAngleSpectrum)

        # Plot-Window Options
        self.window_options_action = createAction('Window Options', self.plot_menu, 'Window parameters',
                                                  self.windowOptionsDialog)

        # Filter
        self.filter_menu = createMenu('Filter', self.menu_bar, enabled=False)

        # Filter-If Update Data
        self.update_data_action = createAction('Update Data(False)', self.filter_menu,
                                               'if True, data will be updated after each filtering '
                                               'operation for continuous process', self.updateFilteredData)

        self.filter_menu.addSeparator()

        # Filter-Detrend
        self.detrend_menu = createMenu('Detrend', self.filter_menu,
                                       status_tip='Remove linear trend along x axis from data')

        # Filter-Detrend-Linear
        self.detrend_linear_action = createAction('Linear', self.detrend_menu,
                                                  'The result of a linear least-squares fit to data '
                                                  'is subtracted from data', self.detrendData)

        # Filter-Detrend-Constant
        self.detrend_constant_action = createAction('Constant', self.detrend_menu, 'The mean of data is subtracted',
                                                    self.detrendData)

        # Filter-EMD
        self.emd_menu = createMenu('EMD', self.filter_menu, status_tip='Use EMD etc. to decompose and reconstruct data')

        # Filter-EMD-EMD, EEMD and CEEMDAN
        emd_method_list = ['EMD', 'EEMD', 'CEEMDAN']
        for i in emd_method_list:
            _ = createAction(i, self.emd_menu, f'Use {i} to decompose and reconstruct data', self.plotEMD)

        self.emd_menu.addSeparator()

        # Filter-EMD-Plot Instantaneous Frequency
        self.emd_plot_ins_fre_action = createAction('Plot Instantaneous Frequency', self.emd_menu,
                                                    'Plots and shows instantaneous frequencies for provided IMF(s)',
                                                    self.plotEMDInstantaneousFrequency)
        self.emd_plot_ins_fre_action.setEnabled(False)

        self.emd_menu.addSeparator()

        # Filter-EMD-Option
        self.emd_options_action = createAction('Options', self.emd_menu, 'EMD options', self.EMDOptionsDialog)

        # Filter-IIR Filter
        self.iir_menu = createMenu('IIR Filter', self.filter_menu)

        # Filter-IIR Filter-Butterworth etc.
        cal_filter_types = ['Butterworth', 'Chebyshev type I', 'Chebyshev type II', 'Elliptic (Cauer)']
        for i in cal_filter_types:
            _ = createAction(i, self.iir_menu, f'Design a {i} filter', self.iirCalculateFilterParams)

        self.iir_menu.addSeparator()

        # Filter-IIR Filter-Bessel/Thomson
        self.iir_bessel_action = createAction('Bessel/Thomson', self.iir_menu, 'Design a Bessel/Thomson filter',
                                              self.iirDesignBesselFilter)

        self.iir_menu.addSeparator()

        # Filter-IIR Filter-notch etc.
        comb_filter_types = ['Notch Digital Filter', 'Peak (Resonant) Digital Filter',
                             'Notching or Peaking Digital Comb Filter']
        for i in comb_filter_types:
            _ = createAction(i, self.iir_menu, f'Design a {i} filter', self.iirDesignCombFilter)

        # Filter-Wavelet
        self.wavelet_menu = createMenu('Wavelet', self.filter_menu)

        # Filter-Wavelet-Discrete Wavelet Transform
        self.wavelet_dwt_action = createAction('Discrete Wavelet Transform', self.wavelet_menu,
                                               'Use discrete wavelet transform to decompose/reconstruct as filtering '
                                               'or get rid of noise', self.waveletDWTDialog)

        # Filter-Wavelet-Denoise
        self.wavelet_threshold_action = createAction('Denoise', self.wavelet_menu,
                                                     'Denoise input data depending on the mode argument',
                                                     self.waveletThresholdDialog)

        self.wavelet_menu.addSeparator()

        # Filter-Wavelet-Wavelet Packets
        self.wavelet_threshold_action = createAction('Wavelet Packets', self.wavelet_menu,
                                                     'Use Wavelet Packets to decompose data into subnodes '
                                                     'and reconstruct from ones needed', self.waveletPacketsDialog)

    def initTabWidget(self):
        """初始化绘图区"""

        # 绘制灰度图
        self.plot_gray_scale_widget = MyPlotWidget('Gray Scale', 'Time(s)', 'Channel', check_mouse=False)
        self.plot_gray_scale_widget.setXRange(0, 5)
        self.plot_gray_scale_widget.setYRange(0, 200)

        # 绘制单通道相位差-时间图
        self.plot_single_channel_time_widget = MyPlotWidget('Phase Difference', 'Time(s)',
                                                            'Phase Difference(rad)',
                                                            grid=True)
        self.plot_single_channel_time_widget.setXRange(0, 5)

        # 绘制频谱图
        self.plot_amplitude_frequency_widget = MyPlotWidget('Amplitude - Frequency', 'Frequency(Hz)',
                                                            'Amplitude', grid=True)
        self.plot_amplitude_frequency_widget.setXRange(0, 500)

        combine_image_widget = QWidget()
        image_vbox = QVBoxLayout()
        image_vbox.addWidget(self.plot_single_channel_time_widget)
        image_vbox.addWidget(self.plot_amplitude_frequency_widget)
        combine_image_widget.setLayout(image_vbox)

        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)  # 设置tab可移动
        self.tab_widget.setStyleSheet('font-size: 15px; font-family: "Times New Roman";')
        self.tab_widget.setTabsClosable(True)  # 设置tab可关闭
        self.tab_widget.tabCloseRequested[int].connect(self.removeTab)
        self.tab_widget.addTab(self.plot_gray_scale_widget, 'Gray Scale')
        self.tab_widget.addTab(combine_image_widget, 'Single')
        self.tab_widget.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tab_widget.tabBar().setTabButton(1, QTabBar.RightSide, None)  # 设置删除按钮消失

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
        file_path_label = Label('File Path')
        self.file_path_line_edit = LineEdit()
        self.file_path_line_edit.setFocusPolicy(Qt.NoFocus)

        change_file_path_button = PushButton('')
        setPicture(file_path_jpg, 'file_path.jpg', change_file_path_button)
        change_file_path_button.clicked.connect(self.changeFilePath)

        file_table_scrollbar = QScrollBar(Qt.Vertical)
        file_table_scrollbar.setStyleSheet('min-height: 100')  # 设置滚动滑块的最小高度
        self.files_table_widget = QTableWidget(100, 1)
        self.files_table_widget.setVerticalScrollBar(file_table_scrollbar)
        self.files_table_widget.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')
        self.files_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.files_table_widget.setHorizontalHeaderLabels(['File'])  # 设置表头
        self.files_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        QTableWidget.resizeRowsToContents(self.files_table_widget)
        QTableWidget.resizeColumnsToContents(self.files_table_widget)  # 设置表格排与列的宽度随内容改变
        self.files_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置一次选中一排内容
        self.files_table_widget.itemPressed.connect(self.selectDataFromTable)

        # 文件区布局
        file_hbox.addWidget(self.file_path_line_edit)
        file_hbox.addWidget(change_file_path_button)
        file_area_vbox.addWidget(file_path_label)
        file_area_vbox.addLayout(file_hbox)
        file_area_vbox.addWidget(self.files_table_widget)

        # 右侧
        # 参数
        sampling_rate_label = Label('Sampling Rate')
        self.sampling_rate_line_edit = OnlyNumLineEdit()
        self.sampling_rate_line_edit.setFocusPolicy(Qt.NoFocus)

        sampling_times_label = Label('Sampling Times')
        self.current_sampling_times_line_edit = OnlyNumLineEdit()
        self.current_sampling_times_line_edit.setFocusPolicy(Qt.NoFocus)

        number_of_channels_label = Label('Number of Channels')
        self.current_channels_line_edit = OnlyNumLineEdit()
        self.current_channels_line_edit.setFocusPolicy(Qt.NoFocus)

        channel_number_label = Label('Channel')
        self.channel_number_spinbx = SpinBox()
        self.channel_number_spinbx.setValue(1)
        self.channel_number_spinbx.setMinimumWidth(100)
        self.channel_number_spinbx.textChanged.connect(self.changeChannelNumber)
        self.channel_number_spinbx.textChanged.connect(self.plotSingleChannelTime)
        self.channel_number_spinbx.textChanged.connect(self.plotAmplitudeFrequency)

        # 播放音频按钮
        self.playBtn = PushButton('')
        setPicture(play_jpg, 'play.jpg', self.playBtn)
        self.playBtn.clicked.connect(self.createWavFile)
        self.playBtn.clicked.connect(self.createPlayer)
        self.playBtn.clicked.connect(self.playBtnChangeState)

        # 停止音频播放按钮
        self.stopBtn = PushButton('')
        setPicture(stop_jpg, 'stop.jpg', self.stopBtn)
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

        # GPS时间组件
        gps_from_label = Label('From')
        gps_to_label = Label('To')
        self.gps_from_line_edit = LineEdit()
        self.gps_from_line_edit.setFocusPolicy(Qt.NoFocus)
        self.gps_to_line_edit = LineEdit()
        self.gps_to_line_edit.setFocusPolicy(Qt.NoFocus)

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
        setPicture(play_jpg, 'play.jpg', self.playBtn)
        self.playBtn.setDisabled(False)
        self.stopBtn.setDisabled(False)  # 设置播放按钮
        self.playerState = False  # 播放器否在播放
        self.hasWavFile = False  # 当前通道是否已创建了音频文件
        self.playerHasMedia = False  # 播放器是否已赋予了文件

        # 通道数
        self.channel_from_num = 1  # 起始通道
        self.channel_to_num = self.current_channels  # 终止通道为当前通道数

        # 采样数
        self.sampling_times_from_num = 1  # 起始采样次数
        self.sampling_times_to_num = self.sampling_times  # 终止采样次数
        self.current_sampling_times = self.sampling_times_to_num - self.sampling_times_from_num + 1  # 当前采样次数

        # 数据单位
        self.data_units = ['phase difference', 'strain rate']  # 相位差与应变率相转化
        self.data_unit_index = 0
        self.strain_rate_action.setChecked(False)
        self.phase_difference_action.setChecked(True)  # 默认输入数据单位为相位差

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
        self.binary_image_threshold_methods = {'Two Peaks': twoPeaks, 'OSTU': OSTU}  # 两种计算方法，双峰法及大津法
        self.binary_image_threshold_method_index = 0  # 计算阈值方法的索引

        # 加窗
        self.window_length = 32  # 窗长
        self.window_text = 'Rectangular / Dirichlet'  # 加窗名称
        self.window_method = boxcar  # 加窗种类
        self.window_overlap_size_ratio = 0.5  # 窗口重叠比
        self.window_overlap_size = int(round(self.window_overlap_size_ratio * self.window_length))  # 默认窗口重叠长度，取整
        self.window_methods = {'Bartlett': bartlett, 'Blackman': blackman, 'Blackman-Harris': blackmanharris,
                               'Bohman': bohman, 'Cosine': cosine, 'Flat Top': flattop,
                               'Hamming': hamming, 'Hann': hann, 'Lanczos / Sinc': lanczos,
                               'Modified Barrtlett-Hann': barthann, 'Nuttall': nuttall, 'Parzen': parzen,
                               'Rectangular / Dirichlet': boxcar, 'Taylor': taylor, 'Triangular': triang,
                               'Tukey / Tapered Cosine': tukey}  # 默认窗口名称及对应的窗口

        # 滤波器是否更新数据
        self.if_update_data = False

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
    """一些特殊函数"""

    def closeEvent(self, event):
        """退出时的提示"""

        reply = QMessageBox.question(self, 'Tip', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        event.accept() if reply == QMessageBox.Yes else event.ignore()

    def removeTab(self, index):
        """关闭对应选项卡"""

        self.tab_widget.removeTab(index)

    # """------------------------------------------------------------------------------------------------------------"""
    """播放当前文件调用的函数"""

    def playBtnChangeState(self):
        """点击播放按钮改变文字和播放器状态"""

        if not self.playerState:
            setPicture(pause_jpg, 'pause.jpg', self.playBtn)
            self.player.play()
            self.playerState = True
        else:
            setPicture(play_jpg, 'play.jpg', self.playBtn)
            self.player.pause()
            self.playerState = False

    def createWavFile(self):
        """创建当前数据的wav文件，储存在当前文件夹路径下"""

        if not self.hasWavFile:
            data = np.array(self.data[self.channel_number])  # 不转array会在重复转换数据类型时发生数据类型错误

            self.temp_wavfile = wave.open(os.path.join(self.file_path, 'temp.wav'), 'wb')  # 在放置数据的文件夹中创建一个临时文件
            self.temp_wavfile.setnchannels(1)  # 设置通道数
            self.temp_wavfile.setsampwidth(2)  # 设置采样宽
            self.temp_wavfile.setframerate(self.sampling_rate)  # 设置采样
            self.temp_wavfile.setnframes(self.current_sampling_times)  # 设置帧数
            self.temp_wavfile.setcomptype('NONE', 'not compressed')  # 设置采样格式  无压缩

            data /= np.max(np.abs(data))  # 归一化至[-1, 1]
            data *= 32768  # 转16位整数必要
            data = data.astype(np.int16).tobytes()  # 转16位整数类型后转比特
            self.temp_wavfile.writeframes(data)
            self.temp_wavfile.close()
            self.hasWavFile = True

    def createPlayer(self):
        """创建播放器并赋数据"""

        if not self.playerHasMedia:
            self.player = QtMultimedia.QMediaPlayer()
            self.player.stateChanged.connect(self.playerStateChanged)
            self.player.setMedia(
                QtMultimedia.QMediaContent(QUrl.fromLocalFile(os.path.join(self.file_path, 'temp.wav'))))
            self.playerHasMedia = True

    def playerStateChanged(self, state):
        """播放器停止后删除文件"""

        if state == QtMultimedia.QMediaPlayer.StoppedState:
            self.resetPlayer()
            os.remove(os.path.join(self.file_path, 'temp.wav'))  # 在播放完成或点击Abort后删除临时文件

    def resetPlayer(self):
        """重置播放器"""

        self.player.stop()
        setPicture(play_jpg, 'play.jpg', self.playBtn)
        self.playerState = False
        self.hasWavFile = False
        self.playerHasMedia = False

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制三个固有图的函数"""

    def plotSingleChannelTime(self):
        """绘制单通道相位差-时间图"""

        data = self.data[self.channel_number - 1]

        self.plot_single_channel_time_widget.plot_item.clear()
        self.plot_single_channel_time_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                                       self.sampling_times_to_num / self.sampling_rate)
        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.data_unit_index == 0:
            self.plot_single_channel_time_widget.setTitle(
                '<font face="Times New Roman" size="5">Phase Difference</font>')
            self.plot_single_channel_time_widget.setLabel('left',
                                                          '<font face="Times New Roman">Phase Difference(rad)</font>')
        else:
            self.plot_single_channel_time_widget.setTitle(
                '<font face="Times New Roman" size="5">Strain Rate</font>')
            self.plot_single_channel_time_widget.setLabel('left',
                                                          '<font face="Times New Roman">Strain Rate(s^-1)</font>')

        self.plot_single_channel_time_widget.draw(x, data, pen=QColor('blue'))

    def plotAmplitudeFrequency(self):
        """绘制幅值-频率图"""

        data = self.data[self.channel_number - 1]

        data = toAmplitude(data, self.current_sampling_times)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        self.plot_amplitude_frequency_widget.plot_item.clear()
        self.plot_amplitude_frequency_widget.setXRange(0, self.sampling_rate / 2)
        y = fixDateLength(self.current_sampling_times)
        self.plot_amplitude_frequency_widget.draw(x, data[:y // 2], pen=QColor('blue'))  # 只要半谱

    def plotGrayChannelsTime(self):
        """绘制灰度 通道-时间图"""

        data = normalizeToGrayScale(self.data)

        tr = QTransform()
        tr.scale(1 / self.sampling_rate, 1)
        tr.translate(self.sampling_times_from_num, 0)

        self.plot_gray_scale_widget.clear()
        self.plot_gray_scale_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                              self.sampling_times_to_num / self.sampling_rate)
        self.plot_gray_scale_widget.setYRange(1, self.current_channels)
        item = pg.ImageItem()
        item.setImage(data.T)
        item.setTransform(tr)  # 将灰度图缩放移动
        self.plot_gray_scale_widget.addItem(item)

    # """------------------------------------------------------------------------------------------------------------"""
    """文件路径区和文件列表调用函数"""

    def changeFilePath(self):
        """更改显示的文件路径"""

        file_path = QFileDialog.getExistingDirectory(self, 'Select File Path', '')  # 起始路径
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
        self.update()

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

        self.plotGrayChannelsTime()
        self.plotSingleChannelTime()
        self.plotAmplitudeFrequency()

    def update(self):
        """总更新函数"""

        self.updateMenuBar()
        self.updateFile()
        self.updateDataParams()
        self.updateDataGPSTime()
        self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""
    """File-Import菜单调用函数"""

    def importData(self):
        """导入（多个）数据文件后更新参数和绘图等"""

        self.file_names = QFileDialog.getOpenFileNames(self, 'Import', '', 'DAS data (*.dat)')[0]  # 打开多个.dat文件
        if self.file_names != []:
            self.file_path = os.path.dirname(self.file_names[0])

            self.readData()
            self.initLocalParams()
            self.update()

    def readData(self):
        """读取数据，更新参数"""

        time, data = [], []
        for file in self.file_names:
            with open(os.path.join(self.file_path, file), 'rb') as f:
                raw_data = np.fromfile(f, dtype='<f4')  # <低位在前高位在后（小端模式），f4：32位（单精度）浮点类型
                sampling_rate = int(raw_data[6])  # 采样率
                sampling_times = int(raw_data[7])  # 采样次数
                channels_num = int(raw_data[9])  # 通道数
                time.append(raw_data[:6])  # GPS时间
                data.append(raw_data[10:].reshape((channels_num, sampling_times)).T)
        data = np.concatenate(data).T  # （通道数，采样次数）

        # 初始化数据相关参数
        self.sampling_rate = sampling_rate
        self.sampling_times = sampling_times * len(self.file_names)
        self.current_channels = channels_num
        self.data, self.origin_data = data, data
        self.time = time

    # """------------------------------------------------------------------------------------------------------------"""
    """File-Export调用函数"""

    def exportData(self):
        """导出数据"""

        fpath, ftype = QFileDialog.getSaveFileName(self, 'Export', '', 'csv(*.csv);;json(*.json);;pickle(*.pickle);;'
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

        self.dialog = QDialog()
        self.dialog.setMaximumWidth(500)
        self.dialog.setWindowTitle('Calculate SNR')

        signal_channel_number_label = Label('Signal Channel Number')
        self.signal_channel_number_line_edit = OnlyNumLineEdit()
        self.signal_channel_number_line_edit.setText(str(self.signal_channel_number))
        signal_start_sampling_time_label = Label('Start Sampling Time')
        self.signal_start_sampling_time_line_edit = OnlyNumLineEdit()
        self.signal_start_sampling_time_line_edit.setText(str(self.signal_start_sampling_time))
        signal_stop_sampling_time_label = Label('Stop Sampling Time')
        self.signal_stop_sampling_time_line_edit = OnlyNumLineEdit()
        self.signal_stop_sampling_time_line_edit.setText(str(self.signal_stop_sampling_time))

        noise_channel_number_label = Label('Noise Channel Number')
        self.noise_channel_number_line_edit = OnlyNumLineEdit()
        self.noise_channel_number_line_edit.setText(str(self.noise_channel_number))
        noise_start_sampling_time_label = Label('Start Sampling Time')
        self.noise_start_sampling_time_line_edit = OnlyNumLineEdit()
        self.noise_start_sampling_time_line_edit.setText(str(self.noise_start_sampling_time))
        noise_stop_sampling_time_label = Label('Stop Sampling Time')
        self.noise_stop_sampling_time_line_edit = OnlyNumLineEdit()
        self.noise_stop_sampling_time_line_edit.setText(str(self.noise_stop_sampling_time))

        snr_label = Label('SNR = ')
        self.snr_line_edit = NumPointLineEdit()
        self.snr_line_edit.setMaximumWidth(100)
        self.snr_line_edit.setFocusPolicy(Qt.NoFocus)
        snr_unit_label = Label('dB')

        btn = PushButton('Calculate')
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
    """Operation-Clip Data By Sampling Times调用函数"""

    def clipSamplingTimesDialog(self):
        """调用裁剪数据对话框"""

        dialog = QDialog()
        dialog.setWindowTitle('Clip Data')

        from_label = Label('From')
        self.sampling_times_from = OnlyNumLineEdit()
        self.sampling_times_from.setText(str(self.sampling_times_from_num))
        to_label = Label('To')
        self.sampling_times_to = OnlyNumLineEdit()
        self.sampling_times_to.setText(str(self.sampling_times_to_num))

        btn = PushButton('OK')
        btn.clicked.connect(self.clipSamplingTimes)
        btn.clicked.connect(self.updateDataParams)
        btn.clicked.connect(self.updateImages)
        btn.clicked.connect(dialog.close)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(from_label)
        hbox.addWidget(self.sampling_times_from)
        hbox.addStretch(1)
        hbox.addWidget(to_label)
        hbox.addWidget(self.sampling_times_to)
        vbox.addLayout(hbox)
        vbox.addWidget(btn)

        dialog.setLayout(vbox)
        dialog.exec_()

    def clipSamplingTimes(self):
        """根据所选采样次数来裁剪数据"""

        from_num, to_num = int(self.sampling_times_from.text()), int(self.sampling_times_to.text())
        list_num = [from_num, to_num]

        # 筛选小数为起始采样次数，大数作为终止采样次数
        if max(list_num) >= self.sampling_times_to_num:
            from_num = min(list_num)
            to_num = self.sampling_times_to_num
        elif min(list_num) <= self.sampling_times_from_num:
            from_num = self.sampling_times_from_num
            to_num = max(list_num)
        else:
            from_num = min(list_num)
            to_num = max(list_num)

        self.sampling_times_from_num = from_num
        self.sampling_times_to_num = to_num

        # 捕获索引错误等
        try:
            self.data = self.origin_data[self.channel_from_num - 1:self.channel_to_num,
                        self.sampling_times_from_num - 1:self.sampling_times_to_num]

        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """Clip Data By Channel调用函数"""

    def clipChannelsDialog(self):
        """按通道裁剪数据"""

        dialog = QDialog()
        dialog.setWindowTitle('Clip Data')

        from_label = Label('From:')
        self.channel_from = OnlyNumLineEdit()
        self.channel_from.setText(str(self.channel_from_num))
        to_label = Label('To:')
        self.channel_to = OnlyNumLineEdit()
        self.channel_to.setText(str(self.channel_to_num))

        btn = PushButton('OK')
        btn.clicked.connect(self.clipChannels)
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

    def clipChannels(self):
        """以通道数裁剪"""

        from_num, to_num= int(self.channel_from.text()), int(self.channel_to.text())
        list_num = [from_num, to_num]

        # 筛选小数为起始通道，大数为终止通道
        if max(list_num) >= self.channel_to_num:
            from_num = min(list_num)
            to_num = self.channel_to_num
        elif min(list_num) <= self.channel_from_num:
            from_num = self.channel_from_num
            to_num = max(list_num)
        else:
            from_num = min(list_num)
            to_num = max(list_num)

        self.channel_from_num = from_num
        self.channel_to_num = to_num

        # 捕获索引错误等
        try:
            self.data = self.origin_data[self.channel_from_num - 1:self.channel_to_num,
                        self.sampling_times_from_num - 1:self.sampling_times_to_num]

        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """转换数据单位调用的函数"""

    def convertToPhaseDifference(self):
        """应变率转为相位差"""

        self.phase_difference_action.setChecked(True)
        self.strain_rate_action.setChecked(False)
        if self.data_unit_index == 1:
            self.data_unit_index = 0
            self.data = convertDataUnit(self.data, self.sampling_rate, src='SR', aim='PD')
            self.updateImages()

    def convertToStrainRate(self):
        """相位差转换为应变率"""

        self.strain_rate_action.setChecked(True)
        self.phase_difference_action.setChecked(False)
        if self.data_unit_index == 0:
            self.data_unit_index = 1
            self.data = convertDataUnit(self.data, self.sampling_rate, src='PD', aim='SR')
            self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""
    """更改读取通道号步长、读取文件数调用的函数"""

    def changeChannelNumberStep(self):
        """改变通道号的步长"""

        dialog = QDialog()
        dialog.setWindowTitle('Change Channel Number Step')

        channel_number_step_label = Label('Channel Number Step')
        self.channel_number_step_line_edit = OnlyNumLineEdit()
        self.channel_number_step_line_edit.setToolTip('Change the step when changing channel number')
        self.channel_number_step_line_edit.setText(str(self.channel_number_step))

        btn = PushButton('OK')
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

        dialog = QDialog()
        dialog.setWindowTitle('Change Files Read Number')

        files_read_number_label = Label('Files Read Number')
        self.files_read_number_line_edit = OnlyNumLineEdit()
        self.files_read_number_line_edit.setToolTip(
            'Number of files read when choosing file from the table, counting from the chosen one.')
        self.files_read_number_line_edit.setText(str(self.files_read_number))

        btn = PushButton('OK')
        btn.clicked.connect(self.updateFilesReadNumber)
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

        self.files_read_number = int(self.files_read_number_line_edit.text())

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制二值图调用函数"""

    def binaryImageDialog(self):
        """二值图设置组件"""

        dialog = QDialog()
        dialog.setWindowTitle('Plot Binary Image')

        self.binary_image_input_radiobtn = RadioButton('Threshold')
        self.binary_image_input_radiobtn.setChecked(self.binary_image_flag)

        self.binary_image_threshold_line_edit = NumPointLineEdit()
        self.binary_image_threshold_line_edit.setText(str(self.binary_image_threshold))

        self.binary_image_method_radiobtn = RadioButton('Method')
        self.binary_image_method_radiobtn.setChecked(not self.binary_image_flag)

        self.binary_image_method_combx = ComboBox()
        self.binary_image_method_combx.addItems(self.binary_image_threshold_methods.keys())
        self.binary_image_method_combx.setCurrentIndex(self.binary_image_threshold_method_index)

        btn = PushButton('OK')
        btn.clicked.connect(self.updateBinaryImageParams)
        btn.clicked.connect(self.plotBinaryImage)
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
            threshold = self.binary_image_threshold_methods[self.binary_image_method_combx.currentText()](self.data)

        self.binary_image_threshold_method_index = self.binary_image_method_combx.currentIndex()
        self.binary_image_threshold = threshold

    def plotBinaryImage(self):
        """绘制二值图"""

        binary_image_widget = MyPlotWidget('Binary Image', 'Time(s)', 'Channel', check_mouse=False)
        self.tab_widget.addTab(binary_image_widget,
                               f'Binary Image - Threshold={self.binary_image_threshold}')  # 添加二值图tab

        # 阈值化
        binary_data = normalizeToGrayScale(self.data)
        binary_data[binary_data >= self.binary_image_threshold] = 255
        binary_data[binary_data < self.binary_image_threshold] = 0  # 根据阈值赋值

        binary_image_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                      self.sampling_times_to_num / self.sampling_rate)
        binary_image_widget.setYRange(1, self.current_channels)

        tr = QTransform()
        tr.scale(1 / self.sampling_rate, 1)
        tr.translate(self.sampling_times_from_num, 0)

        binary_image = pg.ImageItem(binary_data.T)
        binary_image.setTransform(tr)  # 缩放与平移
        binary_image_widget.addItem(binary_image)

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

    def plotFeature(self, features):
        """获取要计算的数据特征名字和值"""

        feature_name = self.plot_data_features_menu.sender().text()
        feature = features[feature_name]

        plot_widget = MyPlotWidget(feature_name, 'Channel', '')
        x = np.linspace(1, self.current_channels, self.current_channels)
        plot_widget.setXRange(1, self.current_channels)
        plot_widget.draw(x, feature, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'{feature_name} - Window Method={self.window_text}')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制多通道云图调用函数"""

    def plotMultiChannelImage(self):
        """绘制多通道云图"""

        plot_widget = MyPlotWidget('MultiChannel Image', 'Time(s)', 'Channel', check_mouse=False)
        plot_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        plot_widget.setYRange(1, self.current_channels)
        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        colors = cycle(['red', 'lime', 'deepskyblue', 'yellow', 'plum', 'gold', 'blue', 'fuchsia', 'aqua', 'orange'])
        for i in range(1, self.current_channels + 1):
            plot_widget.draw(x, self.data[i - 1] + i, pen=QColor(next(colors)))  # 根据通道数个位选择颜色绘图
        self.tab_widget.addTab(plot_widget, 'Channels - Time')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制应变图调用函数"""

    def plotStrain(self):
        """将相位差转为应变率再积分"""

        if self.data_unit_index == 0:
            data = convertDataUnit(self.data, self.sampling_rate, src='PD', aim='S')[self.channel_number]
        else:
            data = convertDataUnit(self.data, self.sampling_rate, src='SR', aim='S')[self.channel_number]

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        plot_widget = MyPlotWidget('Strain Image', 'Time(s)', 'Strain(με)', grid=True)
        plot_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Strain Image - Channel Number={self.channel_number}')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制PSD调用函数"""

    def plotPSD(self):
        """绘制psd图线"""

        data = self.data[self.channel_number - 1]
        data = self.window_method(self.current_sampling_times) * data
        data = np.abs(np.fft.fft(data))
        y = fixDateLength(self.current_sampling_times)
        data = 20.0 * np.log10(data ** 2 / self.current_sampling_times)[:y // 2]  # 转dB单位

        plot_widget = MyPlotWidget('PSD', 'Frequency(Hz)', 'Power/Frequency(dB/Hz)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.setXRange(0, self.sampling_rate / 2)
        plot_widget.setLogMode(x=True)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'PSD - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dPSD(self):
        """绘制2dpsd谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget,
                               f'2D PSD - Window Method={self.window_text}\n'
                               f'Channel Number={self.channel_number}')
        data = self.data[self.channel_number - 1]
        ax = plt.axes()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='density', mode='psd')
        plt.pcolormesh(t, f, 20.0 * np.log10(Sxx), cmap='rainbow')
        plt.colorbar(label='Power/Frequency(dB/Hz)')
        plt.title('2D PSD')
        plt.xlabel('Time(s)')
        plt.ylabel('Frequency(Hz)')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dPSD(self):
        """绘制3dpsd"""

        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget,
                               f'3D PSD - Window Method={self.window_text}\n'
                               f'Channel Number={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='density', mode='psd')
        im = ax.plot_surface(f[:, None], t[None, :], 20.0 * np.log10(Sxx), cmap='rainbow')
        # plt.colorbar(im, ax=ax, label='Power/Frequency(dB/Hz)', pad=0.2)
        ax.set_title('3D PSD')
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('Time(s)')
        ax.set_zlabel('Power/Frequency(dB/Hz)')
        plt.xlim(0, self.sampling_rate / 2)

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制各种谱调用函数"""

    def plotMagnitudeSpectrum(self):
        """绘制幅度谱"""

        data = self.data[self.channel_number - 1]
        y = fixDateLength(self.current_sampling_times)
        data = self.window_method(self.current_sampling_times) * data
        data = 20.0 * np.log10(np.abs(np.fft.fft(data)) / self.current_sampling_times)[:y // 2]

        plot_widget = MyPlotWidget('Magnitude Spectrum', 'Frequency(Hz)', 'Magnitude(dB)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Magnitude Spectrum - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dMagnitudeSpectrum(self):
        """绘制2d幅度谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'2D Magnitude Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        data = self.data[self.channel_number - 1]
        ax = plt.axes()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='magnitude')
        plt.pcolormesh(t, f, 20.0 * np.log10(Sxx), cmap='rainbow')
        plt.colorbar(label='Magnitude(dB)')
        plt.title('2D Magnitude Spectrum')
        plt.xlabel('Time(s)')
        plt.ylabel('Frequency(Hz)')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dMagnitudeSpectrum(self):
        """绘制3d幅度谱"""

        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'3D Magnitude Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='magnitude')
        im = ax.plot_surface(f[:, None], t[None, :], 20.0 * np.log10(Sxx), cmap='rainbow')
        # plt.colorbar(im, ax=ax, label='Magnitude(dB)', pad=0.2)
        ax.set_title('3D Magnitude Spectrum')
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('Time(s)')
        ax.set_zlabel('Magnitude(dB)')
        plt.xlim(0, self.sampling_rate / 2)

    def plotAngleSpectrum(self):
        """绘制相位谱"""

        data = self.data[self.channel_number - 1]
        y = fixDateLength(self.current_sampling_times)
        data = self.window_method(self.current_sampling_times) * data
        data = np.angle(np.fft.fft(data))[:y // 2]

        plot_widget = MyPlotWidget('Angle Spectrum', 'Frequency(Hz)', 'Angle(rad)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.draw(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Angle Spectrum - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dAngleSpectrum(self):
        """绘制2d相位谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'2D Angle Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        data = self.data[self.channel_number - 1]
        ax = plt.axes()
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='angle')
        plt.pcolormesh(t, f, Sxx, cmap='rainbow')
        plt.colorbar(label='Angle(rad)')
        plt.title('2D Angle Spectrum')
        plt.xlabel('Time(s)')
        plt.ylabel('Frequency(Hz)')
        plt.xlim(0, self.current_sampling_times / self.sampling_rate)

    def plot3dAngleSpectrum(self):
        """绘制3d相位谱"""

        data = self.data[self.channel_number - 1]
        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'3D Angle Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        ax = figure.add_subplot(projection='3d')
        ax.tick_params(axis='both', which='both', direction='in')
        f, t, Sxx = spectrogram(data, self.sampling_rate, window=self.window_method(self.window_length, sym=False),
                                nperseg=self.window_length, noverlap=self.window_overlap_size, nfft=self.window_length,
                                scaling='spectrum', mode='angle')
        im = ax.plot_surface(f[:, None], t[None, :], Sxx, cmap='rainbow')
        # plt.colorbar(im, ax=ax, label='Angle(rad)', pad=0.2)
        ax.set_title('3D Angle Spectrum')
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('Time(s)')
        ax.set_zlabel('Angle(rad)')
        plt.xlim(0, self.sampling_rate / 2)

    # """------------------------------------------------------------------------------------------------------------"""
    """Operation-Window Options调用函数"""

    def windowOptionsDialog(self):
        """Window Options调用窗口"""

        dialog = QDialog()
        dialog.setWindowTitle('Window Options')

        window_method_label = Label('Window Method')
        self.window_method_combx = ComboBox()
        self.window_method_combx.addItems(self.window_methods.keys())
        self.window_method_combx.setCurrentText(self.window_text)

        window_length_label = Label('Window Length')
        self.window_length_line_edit = OnlyNumLineEdit()
        self.window_length_line_edit.setText(str(self.window_length))
        self.window_length_line_edit.setToolTip("Usually 2^n")

        window_overlap_size_ratio_label = Label('Window Overlap Size')
        self.window_overlap_size_ratio_line_edit = NumPointLineEdit()
        self.window_overlap_size_ratio_line_edit.setText(str(self.window_overlap_size_ratio))
        self.window_overlap_size_ratio_line_edit.setToolTip('Overlap ratio between adjacent windows, between [0, 1)')

        btn = PushButton('OK')
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

        if self.if_update_data:
            self.update_data_action.setText('Update Data(False)')
            self.if_update_data = False
        else:
            self.update_data_action.setText('Update Data(True)')
            self.if_update_data = True

    def ifUpdateData(self, flag, data):
        """判断是否在滤波之后更新数据"""

        if flag:
            self.origin_data[self.channel_number] = data
            self.data = self.origin_data
            self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""
    """建立相位差和频谱图"""

    def initTwoPlotWidgets(self, data, title):
        """创建返回结合两个pw的Qwidget"""

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.data_unit_index == 0:
            data_widget = MyPlotWidget(f'{title}', 'Time(s)', 'Phase Difference(rad)', grid=True)
        else:
            data_widget = MyPlotWidget(f'{title}', 'Time(s)', 'Strain Rate(s^-1)', grid=True)

        data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        data_widget.draw(x, data, pen=QColor('blue'))

        data = toAmplitude(data, self.current_sampling_times)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        fre_amp_widget = MyPlotWidget('Amplitude - Frequency', 'Frequency(Hz)', 'Amplitude', grid=True)
        fre_amp_widget.setXRange(0, self.sampling_rate / 2)
        y = fixDateLength(self.current_sampling_times)
        fre_amp_widget.draw(x, data[:y // 2], pen=QColor('blue'))

        combine_widget = QWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(data_widget)
        vbox.addWidget(fre_amp_widget)
        combine_widget.setLayout(vbox)

        return combine_widget

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter-Detrend调用函数"""

    def detrendData(self):
        """去趋势"""

        data = self.data[self.channel_number - 1]
        detrend_type = self.detrend_menu.sender().text().lower()
        data = detrend(data, type=detrend_type)

        combine_widget = self.initTwoPlotWidgets(data, 'Detrend')

        self.tab_widget.addTab(combine_widget,
                               f'Detrend - type={detrend_type}\n'
                               f'Channel Number={self.channel_number}')

        self.ifUpdateData(self.if_update_data, data)

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter-EMD调用函数"""

    def plotEMD(self):
        """绘制emd分解图和重构图"""

        self.emd_method = self.emd_menu.sender().text()
        data = self.data[self.channel_number - 1]

        if self.data_unit_index == 1:  # 如果当前单位是应变率，数据较小需乘大一点
            data *= 10e6

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

        if self.data_unit_index == 1:  # 如果当前数据单位是应变率，之前乘大了，先处理到原来的数量级
            self.imfs_res /= 10e6

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
                    pw_time = MyPlotWidget(f'{self.emd_method} Decompose - Time Domain', '', 'IMF1(rad)')
                    pw_fre = MyPlotWidget(f'{self.emd_method} Decompose - Frequency Domain', '', 'IMF1')
                elif i == len(self.imfs_res) - 1:
                    pw_time = MyPlotWidget('', 'Time(s)', f'Residual(rad)')
                    pw_fre = MyPlotWidget('', 'Frequency(Hz)', f'Residual')
                else:
                    pw_time = MyPlotWidget('', '', f'IMF{i + 1}(rad)')
                    pw_fre = MyPlotWidget('', '', f'IMF{i + 1}')

                x_time = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                                     self.current_sampling_times) / self.sampling_rate
                pw_time.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  self.sampling_times_to_num / self.sampling_rate)
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, self.imfs_res[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])  # 设置时域x轴对应

                data = toAmplitude(self.imfs_res[i], self.current_sampling_times)
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(self.current_sampling_times)
                pw_fre.draw(x_fre, data[:y // 2], pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])  # 设置频域x轴对应

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area,
                                   f'{self.emd_method} - Decompose: Number of IMF={self.imfs_res_num - 1}\n'
                                   f'Channel Number={self.channel_number}')
        else:
            reconstruct_imf = [int(i) for i in re.findall('\d+', self.reconstruct_nums)]  # 映射为整数类型
            data = np.zeros(self.imfs_res[0].shape)
            for i in range(len(reconstruct_imf) - 1):
                imf_num = reconstruct_imf[i]
                data += self.imfs_res[imf_num]  # 重构数据

            combine_widget = self.initTwoPlotWidgets(data, self.emd_method + ' Reconstruct')

            self.tab_widget.addTab(combine_widget, f'Reconstruct: Number of IMF={reconstruct_imf}')

            self.ifUpdateData(self.if_update_data, data)

        self.emd_plot_ins_fre_action.setEnabled(True)

    def EMDOptionsDialog(self):
        """使用emd分解合成滤波"""

        dialog = QDialog()
        dialog.resize(600, 200)
        dialog.setWindowTitle('EMD Options')

        shared_options_label = Label('Shared Options')
        shared_options_label.setAlignment(Qt.AlignHCenter)
        self.emd_decompose_radio_btn = RadioButton('Decompose')
        self.emd_decompose_radio_btn.setChecked(self.emd_options_flag)
        imf_num_label = Label('Number of IMF')
        self.emd_decompose_line_edit = OnlyNumLineEdit()
        self.emd_decompose_line_edit.setToolTip('Number of IMF, max to 9')
        self.emd_decompose_line_edit.setText(str(self.imfs_res_num - 1))

        self.emd_reconstruct_radio_btn = RadioButton('Reconstruct')
        self.emd_reconstruct_radio_btn.setChecked(not self.emd_options_flag)
        reconstruct_imf_number_label = Label('IMF(s) to reconstruct')
        self.emd_reconstruct_line_edit = LineEdit()
        self.emd_reconstruct_line_edit.setToolTip(
            'Numbers of IMF to reconstruct, should be separated by symbol(s) like comma and space')
        self.emd_reconstruct_line_edit.setText(str(self.reconstruct_nums))

        if not hasattr(self, 'imfs_res'):
            self.emd_reconstruct_radio_btn.setEnabled(False)
            reconstruct_imf_number_label.setEnabled(False)
            self.emd_reconstruct_line_edit.setEnabled(False)

        eemd_options_label = Label('EEMD Options')
        eemd_options_label.setAlignment(Qt.AlignHCenter)
        eemd_trials_label = Label('Trial Points')
        self.eemd_trials_line_edit = OnlyNumLineEdit()
        self.eemd_trials_line_edit.setText(str(self.eemd_trials))
        self.eemd_trials_line_edit.setToolTip('Number of trials or EMD performance with added noise')
        eemd_noise_width_label = Label('Noise Width')
        self.eemd_noise_width_line_edit = NumPointLineEdit()
        self.eemd_noise_width_line_edit.setText(str(self.eemd_noise_width))
        self.eemd_noise_width_line_edit.setToolTip(
            'Standard deviation of Gaussian noise. It’s relative to absolute amplitude of the signal')

        ceemdan_options_label = Label('CEEMDAN Options')
        ceemdan_options_label.setAlignment(Qt.AlignHCenter)
        ceemdan_trials_label = Label('Trial Points')
        self.ceemdan_trials_line_edit = OnlyNumLineEdit()
        self.ceemdan_trials_line_edit.setText(str(self.ceemdan_trials))
        self.ceemdan_trials_line_edit.setToolTip('Number of trials or EMD performance with added noise')
        ceemdan_epsilon_label = Label('Epsilon')
        self.ceemdan_epsilon_line_edit = NumPointLineEdit()
        self.ceemdan_epsilon_line_edit.setText(str(self.ceemdan_epsilon))
        self.ceemdan_epsilon_line_edit.setToolTip('Scale for added noise which multiply std')
        ceemdan_noise_scale_label = Label('Noise Scale')
        self.ceemdan_noise_scale_line_edit = NumPointLineEdit()
        self.ceemdan_noise_scale_line_edit.setText(str(self.ceemdan_noise_scale))
        self.ceemdan_noise_scale_line_edit.setToolTip('Scale (amplitude) of the added noise')
        ceemdan_noise_kind_label = Label('Noise Kind')
        self.ceemdan_noise_kind_combx = ComboBox()
        self.ceemdan_noise_kind_combx.addItems(['normal', 'uniform'])
        self.ceemdan_noise_kind_combx.setCurrentIndex(self.ceemdan_noise_kind_index)
        ceemdan_range_thr_label = Label('Range Threshold')
        self.ceemdan_range_thr_line_edit = NumPointLineEdit()
        self.ceemdan_range_thr_line_edit.setText(str(self.ceemdan_range_thr))
        self.ceemdan_range_thr_line_edit.setToolTip(
            'Range threshold used as an IMF check. The value is in percentage compared to initial signal’s amplitude. '
            'If absolute amplitude (max - min) is below the range_thr then the decomposition is finished')
        ceemdan_total_power_thr_label = Label('Total Power Threshold')
        self.ceemdan_total_power_thr_line_edit = NumPointLineEdit()
        self.ceemdan_total_power_thr_line_edit.setText(str(self.ceemdan_total_power_thr))
        self.ceemdan_total_power_thr_line_edit.setToolTip(
            'Signal’s power threshold. Finishes decomposition if sum(abs(r)) < thr')

        btn = PushButton('OK')
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

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
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
                pw_list.append(MyPlotWidget('Instantaneous Frequency of IMF(s)', '', 'IMF1(Hz)'))
            elif i == len(inst_freqs) - 1:
                pw_list.append(MyPlotWidget('', 'Time(s)', f'Residual'))
            else:
                pw_list.append(MyPlotWidget('', '', f'IMF{i + 1}(Hz)'))

            pw_list[i].setXRange(self.sampling_times_from_num / self.sampling_rate,
                                 self.sampling_times_to_num / self.sampling_rate)
            pw_list[i].setXLink(pw_list[0])
            pw_list[i].draw(x, inst_freqs[i], pen=QColor('blue'))
            pw_list[i].setFixedHeight(150)
            vbox.addWidget(pw_list[i])
        wgt.setLayout(vbox)
        scroll_area.setWidget(wgt)
        self.tab_widget.addTab(scroll_area, f'EMD - Instantaneous Frequency:\n'
                                            f'Channel Number={self.channel_number}')

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter-IIR Filter-Butterworth调用函数"""

    def iirCalculateFilterParams(self):
        """计算滤波器阶数和自然频率"""

        if not hasattr(self,
                       'filter') or self.filter.filter_name != self.iir_menu.sender().text():  # 如果没有操作过或两次选择的滤波器不同
            self.filter = FilterI(self.iir_menu.sender().text())

            dialog_layout = QVBoxLayout()
            dialog_layout.addLayout(self.filter.cal_vbox)
            dialog_layout.addSpacing(10)
            dialog_layout.addLayout(self.filter.vbox)
            dialog_layout.addSpacing(10)
            dialog_layout.addWidget(self.filter.btn)
            self.filter.dialog.setLayout(dialog_layout)

            self.filter.btn.clicked.connect(self.plotIIRFilter)
            self.filter.btn.clicked.connect(self.filter.dialog.close)

        self.filter.dialog.exec_()

    def iirDesignBesselFilter(self):
        """设计Bessel/Thomson滤波器"""

        if not hasattr(self, 'filter') or self.filter.filter_name != 'Bessel/Thomson':
            self.filter = FilterI('Bessel/Thomson')

            self.filter.btn.clicked.connect(self.plotIIRFilter)
            self.filter.btn.clicked.connect(self.filter.dialog.close)

        self.filter.dialog.exec_()

    def iirDesignCombFilter(self):
        """设计comb类滤波器"""

        if not hasattr(self, 'filter') or self.filter.filter_name != self.iir_menu.sender().text():
            self.filter = FilterII(self.iir_menu.sender().text())

            self.filter.btn.clicked.connect(self.plotIIRFilter)
            self.filter.btn.clicked.connect(self.filter.dialog.close)

        self.filter.dialog.exec_()

    def plotIIRFilter(self):
        """绘制iir滤波器图"""

        data = self.data[self.channel_number - 1]
        data = filtfilt(self.filter.b, self.filter.a, data)  # 滤波

        combine_widget = self.initTwoPlotWidgets(data, 'IIRFilter')

        if hasattr(self.filter, 'method'):
            self.tab_widget.addTab(combine_widget,
                                   f'Filtered Image - Filter={self.filter.filter_name}\n'
                                   f'Method={self.filter.method}\t'
                                   f'Channel Number={self.channel_number}')
        else:
            self.tab_widget.addTab(combine_widget,
                                   f'Filtered Image - Filter={self.filter.filter_name}\n'
                                   f'Channel Number={self.channel_number}')

        self.ifUpdateData(self.if_update_data, data)

    # """------------------------------------------------------------------------------------------------------------"""
    """Wavelet-Discrete Wavelet Transform菜单调用函数"""

    def waveletDWTDialog(self):
        """设置选择的小波、分解层数、填充模式"""

        dialog = QDialog()
        dialog.setWindowTitle('Discrete Wavelet Transform')

        self.wavelet_dwt_decompose_radiobtn = RadioButton('Decompose')
        self.wavelet_dwt_decompose_radiobtn.setChecked(self.wavelet_dwt_flag)

        wavelet_dwt_reconstruct_radiobtn = RadioButton('Reconstruct')
        wavelet_dwt_reconstruct_radiobtn.setChecked(not self.wavelet_dwt_flag)

        wavelet_dwt_reconstruct_label = Label('Coefficient(s) to reconstruct')
        self.wavelet_dwt_reconstruct_line_edit = LineEdit()
        self.wavelet_dwt_reconstruct_line_edit.setFixedWidth(500)
        self.wavelet_dwt_reconstruct_line_edit.setToolTip(
            'Choose cAn and cDn to reconstruct\ncAn: approximation coefficients\ncDn-cD1: detail coefficients'
            'just delete the unwanted cDn(s) and leave the format as they were')
        self.wavelet_dwt_reconstruct_line_edit.setText(str(self.wavelet_dwt_reconstruct))

        if not hasattr(self, 'wavelet_dwt_coeffs'):
            wavelet_dwt_reconstruct_radiobtn.setEnabled(False)
            wavelet_dwt_reconstruct_label.setEnabled(False)
            self.wavelet_dwt_reconstruct_line_edit.setEnabled(False)

        wavelet_dwt_label = Label('Wavelet To Use:')

        wavelet_dwt_family_label = Label('Wavelet Family')
        self.wavelet_dwt_family_combx = ComboBox()
        self.wavelet_dwt_family_combx.addItems(pywt.families(short=False))
        self.wavelet_dwt_family_combx.setCurrentIndex(self.wavelet_dwt_family_index)
        self.wavelet_dwt_family_combx.currentIndexChanged[int].connect(self.waveletDWTChangeNameComboBox)

        wavelet_dwt_name_label = Label('Name')
        self.wavelet_dwt_name_combx = ComboBox()
        self.wavelet_dwt_name_combx.setFixedWidth(75)
        self.wavelet_dwt_name_combx.addItems(pywt.wavelist(pywt.families()[self.wavelet_dwt_family_index]))
        self.wavelet_dwt_name_combx.setCurrentIndex(self.wavelet_dwt_name_index)

        wavelet_dwt_decompose_level_label = Label('Decompose Level')
        self.wavelet_dwt_decompose_level_line_edit = OnlyNumLineEdit()
        self.wavelet_dwt_decompose_level_line_edit.setToolTip(
            'Decomposition level, which must be an integer bigger than 0')
        self.wavelet_dwt_decompose_level_line_edit.setText(str(self.wavelet_dwt_decompose_level))

        self.wavelet_dwt_decompose_level_checkbx = CheckBox('Use Calculated Level')
        self.wavelet_dwt_decompose_level_checkbx.setToolTip(
            'If using calculated level, which is determined by data length and chosen wavelet')
        self.wavelet_dwt_decompose_level_checkbx.setChecked(self.wavelet_dwt_decompose_level_calculated)

        wavelet_dwt_padding_mode_label = Label('Padding Mode')
        self.wavelet_dwt_padding_mode_combx = ComboBox()
        self.wavelet_dwt_padding_mode_combx.setToolTip('Signal extension mode')
        self.wavelet_dwt_padding_mode_combx.addItems(pywt.Modes.modes)
        self.wavelet_dwt_padding_mode_combx.setCurrentIndex(self.wavelet_dwt_padding_mode_index)

        btn = PushButton('OK')
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

    def waveletDWTChangeNameComboBox(self, index):
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
        if self.data_unit_index == 1:
            data *= 10e6

        if self.wavelet_dwt_decompose_level_calculated:
            self.wavelet_dwt_decompose_level = pywt.dwt_max_level(self.current_sampling_times,
                                                                  self.wavelet_dwt_name_combx.currentText())  # 求最大分解层数

        if self.wavelet_dwt_flag:
            try:
                self.wavelet_dwt_coeffs = pywt.wavedec(data, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                                       mode=self.wavelet_dwt_padding_mode_combx.currentText(),
                                                       level=self.wavelet_dwt_decompose_level)  # 求分解系数

            except Exception as err:
                printError(err)

            self.wavelet_dwt_reconstruct = []
            self.wavelet_dwt_reconstruct.append(f'cA{self.wavelet_dwt_decompose_level}')
            for i in range(len(self.wavelet_dwt_coeffs) - 1, 0, -1):
                self.wavelet_dwt_reconstruct.append(f'cD{i}')
            self.wavelet_dwt_former_reconstruct = self.wavelet_dwt_reconstruct

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

    def plotWaveletDWT(self, coeffs):
        """绘图"""

        if self.data_unit_index == 1:
            for i in coeffs:
                i /= 10e6

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
                    pw_time = MyPlotWidget('DWT Decompose - Time Domain', '',
                                           f'cA{self.wavelet_dwt_decompose_level}(rad)')
                    pw_fre = MyPlotWidget('DWT Decompose - Frequency Domain', '',
                                          f'cA{self.wavelet_dwt_decompose_level}')
                elif i == len(coeffs) - 1:
                    pw_time = MyPlotWidget('', 'Time(s)', f'cD1(rad)')
                    pw_fre = MyPlotWidget('', 'Frequency(Hz)', f'cD1')
                else:
                    pw_time = MyPlotWidget('', '', f'cD{len(coeffs) - i}(rad)')
                    pw_fre = MyPlotWidget('', '', f'cD{len(coeffs) - i}')

                x_time = np.linspace(self.sampling_times_from_num, self.sampling_times_from_num + len(coeffs[i]),
                                     len(coeffs[i])) / self.sampling_rate
                pw_time.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  (self.sampling_times_from_num + len(coeffs[i])) / self.sampling_rate)
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, coeffs[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])

                data = toAmplitude(coeffs[i], len(coeffs[i]))
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / len(coeffs[i]))
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(len(coeffs[i]))
                pw_fre.draw(x_fre, data[:y // 2], pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area,
                                   f'DWT - Decompose: Level={self.wavelet_dwt_decompose_level}\n'
                                   f'Wavelet={self.wavelet_dwt_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')
        else:
            try:
                data = pywt.waverec(coeffs, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                    mode=self.wavelet_dwt_padding_mode_combx.currentText())  # 重构信号

            except Exception as err:
                printError(err)

            combine_widget = self.initTwoPlotWidgets(data, 'DWT Reconstruct')

            self.tab_widget.addTab(combine_widget,
                                   f'DWT - Reconstruct: Coefficient={self.wavelet_dwt_reconstruct}\n'
                                   f'Wavelet={self.wavelet_dwt_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')

            self.ifUpdateData(self.if_update_data, data)

    def waveletThresholdDialog(self):
        """小波去噪"""

        dialog = QDialog()
        dialog.setWindowTitle('Wavelet Denoise')

        wavelet_threshold_label = Label('Threshold')
        self.wavelet_threshold_line_edit = NumPointLineEdit()
        self.wavelet_threshold_line_edit.setToolTip('Thresholding value')
        self.wavelet_threshold_line_edit.setText(str(self.wavelet_threshold))

        wavelet_threshold_sub_label = Label('Substitute value')
        self.wavelet_threshold_sub_line_edit = NumPointLineEdit()
        self.wavelet_threshold_sub_line_edit.setToolTip('Substitute value')
        self.wavelet_threshold_sub_line_edit.setText(str(self.wavelet_threshold_sub))

        wavelet_threshold_mode_label = Label('Threshold Type')
        self.wavelet_threshold_mode_combx = ComboBox()
        self.wavelet_threshold_mode_combx.setToolTip('Decides the type of thresholding to be applied on input data')
        self.wavelet_threshold_mode_combx.addItems(self.wavelet_threshold_modes)
        self.wavelet_threshold_mode_combx.setCurrentIndex(self.wavelet_threshold_mode_index)

        btn = PushButton('OK')
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

        except Exception as err:
            printError(err)

        combine_widget = self.initTwoPlotWidgets(data, 'Wavelet Threshold')

        self.tab_widget.addTab(combine_widget,
                               f'Wavelet Thresholded Image - Threshold={self.wavelet_threshold}\n'
                               f'Threshold Type={self.wavelet_threshold_mode_combx.currentText()}\t'
                               f'Channel Number={self.channel_number}')

        self.ifUpdateData(self.if_update_data, data)

    def waveletPacketsDialog(self):
        """小波包分解"""

        dialog = QDialog()
        dialog.setWindowTitle('Wavelet Packets')

        self.wavelet_packets_decompose_radiobtn = RadioButton('Decompose')
        self.wavelet_packets_decompose_radiobtn.setChecked(self.wavelet_packets_flag)

        wavelet_packets_reconstruct_radiobtn = RadioButton('Reconstruct')
        wavelet_packets_reconstruct_radiobtn.setChecked(not self.wavelet_packets_flag)

        wavelet_packets_reconstruct_label = Label('Subnode(s) to reconstruct')
        self.wavelet_packets_reconstruct_line_edit = LineEdit()
        self.wavelet_packets_reconstruct_line_edit.setFixedWidth(500)
        self.wavelet_packets_reconstruct_line_edit.setToolTip(
            'Choose a(s) and d(s) to reconstruct\na: approximation coefficients\nd: detail coefficients'
            'just delete the unwanted subnode(s) and leave the format as they were\n'
            'multiple connected a(s) and d(s) represent path of the subnode')
        self.wavelet_packets_reconstruct_line_edit.setText(str(self.wavelet_packets_reconstruct))

        if not hasattr(self, 'wavelet_packets_subnodes'):
            wavelet_packets_reconstruct_radiobtn.setEnabled(False)
            wavelet_packets_reconstruct_label.setEnabled(False)
            self.wavelet_packets_reconstruct_line_edit.setEnabled(False)

        wavelet_packets_label = Label('Wavelet To Use:')

        wavelet_packets_family_label = Label('Wavelet Family')
        self.wavelet_packets_family_combx = ComboBox()
        self.wavelet_packets_family_combx.addItems(pywt.families(short=False))
        self.wavelet_packets_family_combx.setCurrentIndex(self.wavelet_packets_family_index)
        self.wavelet_packets_family_combx.currentIndexChanged[int].connect(self.waveletPacketsChangeNameComboBox)

        wavelet_packets_name_label = Label('Name')
        self.wavelet_packets_name_combx = ComboBox()
        self.wavelet_packets_name_combx.setFixedWidth(75)
        self.wavelet_packets_name_combx.addItems(pywt.wavelist(pywt.families()[self.wavelet_packets_family_index]))
        self.wavelet_packets_name_combx.setCurrentIndex(self.wavelet_packets_name_index)
        self.wavelet_packets_name_combx.currentIndexChanged.connect(self.waveletPacketsCalculateDecomposeMaxLevel)

        wavelet_packets_decompose_level_label = Label('Decompose Level')
        self.wavelet_packets_decompose_level_line_edit = OnlyNumLineEdit()
        self.wavelet_packets_decompose_level_line_edit.setToolTip(
            'Decomposition level, which must be an integer bigger than 0')
        self.wavelet_packets_decompose_level_line_edit.setText(str(self.wavelet_packets_decompose_level))

        wavelet_packets_decompose_max_level_label = Label('Maximum Decompose Level')
        self.wavelet_packets_decompose_max_level_line_edit = OnlyNumLineEdit()
        self.wavelet_packets_decompose_max_level_line_edit.setFocusPolicy(Qt.NoFocus)
        self.wavelet_packets_decompose_max_level_line_edit.setToolTip(
            'Maximum level data can be decomposed, which is determined according to the data length and chosen wavelet')
        self.wavelet_packets_decompose_max_level = pywt.dwt_max_level(self.data.shape[1],
                                                                      self.wavelet_packets_name_combx.currentText())
        self.wavelet_packets_decompose_max_level_line_edit.setText(str(self.wavelet_packets_decompose_max_level))

        wavelet_packets_padding_mode_label = Label('Padding Mode')
        self.wavelet_packets_padding_mode_combx = ComboBox()
        self.wavelet_packets_padding_mode_combx.setToolTip('Signal extension mode')
        self.wavelet_packets_padding_mode_combx.addItems(pywt.Modes.modes)
        self.wavelet_packets_padding_mode_combx.setCurrentIndex(self.wavelet_packets_padding_mode_index)

        btn = PushButton('OK')
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

    def waveletPacketsChangeNameComboBox(self, index):
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
        if self.data_unit_index == 1:
            data *= 10e6

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

        except Exception as err:
            printError(err)

        self.plotWaveletPackets(self.wavelet_packets_subnodes)

    def plotWaveletPackets(self, subnodes):
        """绘图"""

        if self.data_unit_index == 1:
            for i in subnodes:
                i.data /= 10e6

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
                    pw_time = MyPlotWidget('Wavelet Packets Decompose - Time Domain', '',
                                           f'{self.wavelet_packets_subnodes[i].path}(rad)')
                    pw_fre = MyPlotWidget('Wavelet Packets Decompose - Frequency Domain', '',
                                          f'{self.wavelet_packets_subnodes[i].path}')
                elif i == len(subnodes) - 1:
                    pw_time = MyPlotWidget('', 'Time(s)', f'{self.wavelet_packets_subnodes[i].path}(rad)')
                    pw_fre = MyPlotWidget('', 'Frequency(Hz)', f'{self.wavelet_packets_subnodes[i].path}')
                else:
                    pw_time = MyPlotWidget('', '', f'{self.wavelet_packets_subnodes[i].path}(rad)')
                    pw_fre = MyPlotWidget('', '', f'{self.wavelet_packets_subnodes[i].path}')

                x_time = np.linspace(self.sampling_times_from_num, self.sampling_times_from_num + len(subnodes[i].data),
                                     len(subnodes[i].data)) / self.sampling_rate
                pw_time.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  (self.sampling_times_from_num + len(subnodes[i].data)) / self.sampling_rate)
                pw_time.setFixedHeight(150)
                pw_time.draw(x_time, subnodes[i].data, pen=QColor('blue'))
                pw_time_list.append(pw_time)
                pw_time_list[i].setXLink(pw_time_list[0])

                data = toAmplitude(subnodes[i].data, len(subnodes[i].data))
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / len(subnodes[i].data))
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(len(subnodes[i].data))
                pw_fre.draw(x_fre, data[:y // 2], pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area,
                                   f'Wavelet Packets - Decompose: Level={self.wavelet_packets_decompose_level}\n'
                                   f'Wavelet={self.wavelet_packets_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')
        else:
            try:
                data = self.wavelet_packets_wp.reconstruct()  # 重构信号

            except Exception as err:
                printError(err)

            combine_widget = self.initTwoPlotWidgets(data, 'Wavelet Packets Reconstruct')

            self.tab_widget.addTab(combine_widget,
                                   f'Wavelet Packets - Reconstruct: Subnodes={self.wavelet_packets_reconstruct}\n'
                                   f'Wavelet={self.wavelet_packets_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')

            self.ifUpdateData(self.if_update_data, data)

    # """------------------------------------------------------------------------------------------------------------"""


if __name__ == '__main__':
    DAS_Visualizer = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(DAS_Visualizer.exec_())
