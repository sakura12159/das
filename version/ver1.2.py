"""
2023-5-31
ver1.2

1.添加部分功能错误捕捉及提示
2.修复归一化函数错误
3.修复计算滤波器阶数调用函数错误
4.移除了对单道数据整体加窗，现在加窗设置只用于绘制psd、幅度谱和相位谱
5.修复重写组件正则错误
6.调整部分滤波器窗口布局
7.修复导出为.xls和.xlsx时的错误
"""

import ctypes
import sys

import pandas as pd
import pywt
from PyEMD import EMD, EEMD, CEEMDAN
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, qApp, QTabWidget, QTableWidget, \
    QAbstractItemView, QTableWidgetItem, QHeaderView, QTabBar, QWidget, QScrollArea, QScrollBar
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import hilbert, filtfilt, spectrogram
from scipy.signal.windows import *

from en.filter_en import *
from en.function_en import *
from en.image import *
from en.widget_en import *


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initMainWindow()
        self.initUI()
        self.initMenu()
        self.initImage()
        self.initLayout()
        self.initSetting()

    def initMainWindow(self):
        """获取屏幕分辨率，设置主窗口初始大小"""

        self.screen = QApplication.desktop()
        self.screen_height = int(self.screen.screenGeometry().height() * 0.8)
        self.screen_width = int(self.screen.screenGeometry().width() * 0.8)
        self.resize(self.screen_width, self.screen_height)

    def initUI(self):
        """初始化ui"""

        # self.setFont(QFont('Times New Roman', 9))

        self.status_bar = self.statusBar()  # 状态栏
        self.status_bar.setStyleSheet('font-size: 15px; font-family: "Times New Roman";')
        self.menu_bar = self.menuBar()  # 菜单栏
        self.menu_bar.setStyleSheet('font-size: 18px; font-family: "Times New Roman";')
        self.setWindowTitle('DAS Visualizer')

        getPicture(icon_jpg, 'icon.jpg')  # 从image.py中获取图片信息生成图片
        self.setWindowIcon(QIcon('icon.jpg'))  # 加载图片
        os.remove('icon.jpg')  # 移除图片释放内存

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('myappid')  # 设置任务栏图标

    def initMenu(self):
        """初始化菜单"""

        # 菜单栏中的File选项
        file_menu = self.menu_bar.addMenu('File')

        # Import子菜单
        import_action = QAction('Import', file_menu)
        import_action.setStatusTip('Import data file(s)')
        import_action.setShortcut('Ctrl+I')
        import_action.triggered.connect(self.importData)
        file_menu.addAction(import_action)

        # File-Export
        self.export_action = QAction('Export', file_menu)
        self.export_action.setEnabled(False)
        self.export_action.setStatusTip('Export data')
        self.export_action.setShortcut('Ctrl+E')
        self.export_action.triggered.connect(self.exportData)
        file_menu.addAction(self.export_action)

        file_menu.addSeparator()

        # Quit子菜单
        quit_action = QAction('Quit', file_menu)
        quit_action.setStatusTip('Quit DAS_Visualizer')
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(qApp.quit)
        file_menu.addAction(quit_action)

        # 菜单栏中的Operation选项
        self.operation_menu = self.menu_bar.addMenu('Operation')
        self.operation_menu.setEnabled(False)

        # 按采样次数裁剪数据范围
        clip_time_action = QAction('Clip Data By Sampling Times', self.operation_menu)
        clip_time_action.setStatusTip('Clip data within a certain range of sampling times')
        clip_time_action.triggered.connect(self.clipSamplingTimesDialog)
        self.operation_menu.addAction(clip_time_action)

        # 按通道数裁剪数据
        clip_channel_action = QAction('Clip Data By Channels', self.operation_menu)
        clip_channel_action.setStatusTip('Clip data within a certain range of channels')
        clip_channel_action.triggered.connect(self.clipChannelsDialog)
        self.operation_menu.addAction(clip_channel_action)

        self.operation_menu.addSeparator()

        # Operation-Convert...
        convert_menu = QMenu('Convert Data Unit To', self.operation_menu)
        self.operation_menu.addMenu(convert_menu)

        # Operation-Convert...-Phase Difference
        self.phase_difference_action = QAction('Phase Difference(rad)', convert_menu)
        self.phase_difference_action.setStatusTip('Set phase difference as data unit')
        self.phase_difference_action.triggered.connect(self.convert2PhaseDifference)
        self.phase_difference_action.setCheckable(True)
        self.phase_difference_action.setChecked(True)
        convert_menu.addAction(self.phase_difference_action)

        # Operation-Convert...-Strain Rate
        self.strain_rate_action = QAction('Strain Rate(s^-1)', convert_menu)
        self.strain_rate_action.setStatusTip('Set strain rate as data unit')
        self.strain_rate_action.triggered.connect(self.convert2StrainRate)
        self.strain_rate_action.setCheckable(True)
        convert_menu.addAction(self.strain_rate_action)

        self.operation_menu.addSeparator()

        # Operation- Files Read Number
        number_of_files_read_action = QAction('Files Read Number', self.operation_menu)
        number_of_files_read_action.setStatusTip(
            'Number of files read when choosing file from the table, counting from the chosen one.')
        number_of_files_read_action.triggered.connect(self.filesReadNumberDialog)
        self.operation_menu.addAction(number_of_files_read_action)

        # Plot菜单
        self.plot_menu = self.menu_bar.addMenu('Plot')
        self.plot_menu.setEnabled(False)

        # 绘制二值图子菜单
        plot_binary_image_action = QAction('Plot Binary Image', self.plot_menu)
        plot_binary_image_action.setStatusTip('Set the threshold to plot binary image')
        plot_binary_image_action.triggered.connect(self.binaryImageDialog)
        self.plot_menu.addAction(plot_binary_image_action)

        # 计算数据特征
        self.plot_data_features_menu = QMenu('Plot Data Features', self.plot_menu)
        self.plot_data_features_menu.setStatusTip(
            'Calculate and plot time-domain & frequency-domain features of data')
        self.plot_menu.addMenu(self.plot_data_features_menu)

        # 计算数据特征中的各子菜单
        self.time_domain_chars_text = {'max_value': 'Maximum Value', 'peak_value': 'Peak Value',
                                       'min_value': 'Minimum Value', 'mean': 'Mean Value',
                                       'peak_peak_value': 'Peak-To-Peak Value',
                                       'mean_absolute_value': 'Mean-Absolute Value',
                                       'root_mean_square': 'Root-Mean-Square',
                                       'square_root_amplitude': 'Square-Root-Amplitude', 'variance': 'Variance',
                                       'standard_deviation': 'Standard-Deviation', 'kurtosis': 'Kurtosis',
                                       'skewness': 'Skewness', 'clearance_factor': 'Clearance Factor',
                                       'shape_factor': 'Shape Factor', 'impulse_factor': 'Impulse Factor',
                                       'crest_factor': 'Crest Factor', 'kurtosis_factor': 'Kurtosis Factor'}

        self.fre_domain_chars_text = {'centroid_frequency': 'Centroid Frequency', 'mean_frequency': 'Mean Frequency',
                                      'root_mean_square_frequency': 'Root-Mean-Square Frequency',
                                      'frequency_variance': 'Frequency Variance',
                                      'mean_square_frequency': 'Mean-Square Frequency',
                                      'frequency_standard_deviation': 'Frequency Standard-Deviation'}

        for i in self.time_domain_chars_text.values():
            action = QAction(f'{i}', self.plot_data_features_menu)
            action.triggered.connect(self.plotTimeDomainFeature)
            self.plot_data_features_menu.addAction(action)

        # 时域和频域特征之间的分隔线
        self.plot_data_features_menu.addSeparator()

        for i in self.fre_domain_chars_text.values():
            action = QAction(f'{i}', self.plot_data_features_menu)
            action.triggered.connect(self.plotFrequencyDomainFeature)
            self.plot_data_features_menu.addAction(action)

        # Plot-Plot Strain Image
        self.plot_strain_image_action = QAction('Plot Strain Image', self.plot_menu)
        self.plot_strain_image_action.setStatusTip('Convert phase difference to strain')
        self.plot_strain_image_action.triggered.connect(self.plotStrain)
        self.plot_menu.addAction(self.plot_strain_image_action)

        self.plot_menu.addSeparator()

        # Operation-Plot PSD
        plot_psd_menu = QMenu('Plot PSD', self.plot_menu)
        self.plot_menu.addMenu(plot_psd_menu)

        # Operation-Plot PSD-Plot PSD
        plot_psd_action = QAction('Plot PSD', plot_psd_menu)
        plot_psd_action.setStatusTip('Plot psd')
        plot_psd_action.triggered.connect(self.plotPSD)
        plot_psd_menu.addAction(plot_psd_action)

        # Operation-Plot PSD-Plot 2D PSD
        plot_2d_psd_action = QAction('Plot 2D PSD', plot_psd_menu)
        plot_2d_psd_action.setStatusTip('Plot 2d psd')
        plot_2d_psd_action.triggered.connect(self.plot2dPSD)
        plot_psd_menu.addAction(plot_2d_psd_action)

        # Operation-Plot PSD-Plot 3D PSD
        plot_3d_psd_action = QAction('Plot 3D PSD', plot_psd_menu)
        plot_3d_psd_action.setStatusTip('Plot 3d psd')
        plot_3d_psd_action.triggered.connect(self.plot3dPSD)
        plot_psd_menu.addAction(plot_3d_psd_action)

        # Operation-Plot Spectrum
        plot_spectrum_menu = QMenu('Plot Spectrum', self.plot_menu)
        self.plot_menu.addMenu(plot_spectrum_menu)

        # Operation-Plot Spectrum-Plot Magnitude Spectrum
        plot_mag_spectrum_action = QAction('Plot Magnitude Spectrum', plot_spectrum_menu)
        plot_mag_spectrum_action.setStatusTip('Plot magnitude spectrum')
        plot_mag_spectrum_action.triggered.connect(self.plotMagnitudeSpectrum)
        plot_spectrum_menu.addAction(plot_mag_spectrum_action)

        # Operation-Plot Spectrum-Plot 2D Magnitude Spectrum
        plot_2d_mag_spectrum_action = QAction('Plot 2D Magnitude Spectrum', plot_spectrum_menu)
        plot_2d_mag_spectrum_action.setStatusTip('Plot 2d magnitude spectrum')
        plot_2d_mag_spectrum_action.triggered.connect(self.plot2dMagnitudeSpectrum)
        plot_spectrum_menu.addAction(plot_2d_mag_spectrum_action)

        # Operation-Plot Spectrum-Plot 3D Magnitude Spectrum
        plot_3d_mag_psd_action = QAction('Plot 3D Magnitude Spectrum', plot_spectrum_menu)
        plot_3d_mag_psd_action.setStatusTip('Plot 3d magnitude spectrum')
        plot_3d_mag_psd_action.triggered.connect(self.plot3dMagnitudeSpectrum)
        plot_spectrum_menu.addAction(plot_3d_mag_psd_action)

        plot_spectrum_menu.addSeparator()

        # Operation-Plot Spectrum-Plot Angle Spectrum
        plot_ang_spectrum_action = QAction('Plot Angle Spectrum', plot_spectrum_menu)
        plot_ang_spectrum_action.setStatusTip('Plot angle spectrum')
        plot_ang_spectrum_action.triggered.connect(self.plotAngleSpectrum)
        plot_spectrum_menu.addAction(plot_ang_spectrum_action)

        # Operation-Plot Spectrum-Plot 2D Angle Spectrum
        plot_2d_ang_spectrum_action = QAction('Plot 2D Angle Spectrum', plot_spectrum_menu)
        plot_2d_ang_spectrum_action.setStatusTip('Plot 2d angle spectrum')
        plot_2d_ang_spectrum_action.triggered.connect(self.plot2dAngleSpectrum)
        plot_spectrum_menu.addAction(plot_2d_ang_spectrum_action)

        # Operation-Plot Spectrum-Plot 3D Angle Spectrum
        plot_3d_ang_spectrum_action = QAction('Plot 3D Angle Spectrum', plot_spectrum_menu)
        plot_3d_ang_spectrum_action.setStatusTip('Plot 3d angle spectrum')
        plot_3d_ang_spectrum_action.triggered.connect(self.plot3dAngleSpectrum)
        plot_spectrum_menu.addAction(plot_3d_ang_spectrum_action)

        # Operation-Window Options 子菜单
        window_options_action = QAction('Window Options', self.plot_menu)
        window_options_action.setStatusTip('Window parameters')
        window_options_action.triggered.connect(self.windowOptionsDialog)
        self.plot_menu.addAction(window_options_action)

        # 菜单栏中的Filter菜单
        self.filter_menu = self.menu_bar.addMenu('Filter')
        self.filter_menu.setEnabled(False)

        # Filter-If Update Data
        self.update_data_action = QAction('Update Data(False)', self.filter_menu)
        self.update_data_action.setStatusTip(
            'if True, data will be updated after each filtering operation for continuous process')
        self.update_data_action.triggered.connect(self.updateFilteredData)
        self.filter_menu.addAction(self.update_data_action)

        self.filter_menu.addSeparator()

        # Filter-EMD
        self.emd_menu = QMenu('EMD', self.filter_menu)
        self.emd_menu.setStatusTip('Use EMD etc. to decompose and reconstruct data')
        self.filter_menu.addMenu(self.emd_menu)

        # Filter-EMD-EMD
        emd_emd_action = QAction('EMD', self.emd_menu)
        emd_emd_action.setStatusTip('Use EMD to decompose and reconstruct data')
        emd_emd_action.triggered.connect(self.plotEMD)
        self.emd_menu.addAction(emd_emd_action)

        # Filter-EMD-EEMD
        emd_eemd_action = QAction('EEMD', self.emd_menu)
        emd_eemd_action.setStatusTip('Use EEMD to decompose and reconstruct data')
        emd_eemd_action.triggered.connect(self.plotEMD)
        self.emd_menu.addAction(emd_eemd_action)

        # Filter-EMD-CEEMDAN
        emd_ceemdan_action = QAction('CEEMDAN', self.emd_menu)
        emd_ceemdan_action.setStatusTip('Use CEEMDAN to decompose and reconstruct data')
        emd_ceemdan_action.triggered.connect(self.plotEMD)
        self.emd_menu.addAction(emd_ceemdan_action)

        self.emd_menu.addSeparator()

        # Filter-EMD-Plot Instantaneous Frequency
        self.emd_plot_ins_fre_action = QAction('Plot Instantaneous Frequency')
        self.emd_plot_ins_fre_action.setEnabled(False)
        self.emd_plot_ins_fre_action.setStatusTip('Plots and shows instantaneous frequencies for provided IMF(s).')
        self.emd_plot_ins_fre_action.triggered.connect(self.plotEMDInstantaneousFrequency)
        self.emd_menu.addAction(self.emd_plot_ins_fre_action)

        self.emd_menu.addSeparator()

        # Filter-EMD-Option
        emd_options_action = QAction('Options', self.emd_menu)
        emd_options_action.setStatusTip('EMD options')
        emd_options_action.triggered.connect(self.EMDOptionsDialog)
        self.emd_menu.addAction(emd_options_action)

        # Filter-IIR Filter
        self.iir_menu = QMenu('IIR Filter', self.filter_menu)
        self.filter_menu.addMenu(self.iir_menu)

        # Filter-IIR Filter-Butterworth etc.
        cal_filter_types = ['Butterworth', 'Chebyshev type I', 'Chebyshev type II', 'Elliptic (Cauer)']
        for i in cal_filter_types:
            action = QAction(i, self.iir_menu)
            action.setStatusTip(f'Design a {i} filter')
            action.triggered.connect(self.iirCalculateFilterParams)
            self.iir_menu.addAction(action)

        self.iir_menu.addSeparator()

        # Filter-IIR Filter-Bessel/Thomson
        iir_bessel_action = QAction('Bessel/Thomson', self.iir_menu)
        iir_bessel_action.setStatusTip('Design a Bessel/Thomson filter')
        iir_bessel_action.triggered.connect(self.iirDesignBesselFilter)
        self.iir_menu.addAction(iir_bessel_action)

        self.iir_menu.addSeparator()

        # Filter-IIR Filter-notch etc.
        comb_filter_types = ['Notch Digital Filter', 'Peak (Resonant) Digital Filter',
                             'Notching or Peaking Digital Comb Filter']
        for i in comb_filter_types:
            action = QAction(i, self.iir_menu)
            action.setStatusTip(f'Design a {i} filter')
            action.triggered.connect(self.iirDesignCombFilter)
            self.iir_menu.addAction(action)

        # Filter-Wavelet
        wavelet_menu = QMenu('Wavelet', self.filter_menu)
        self.filter_menu.addMenu(wavelet_menu)

        # Filter-Wavelet-Discrete Wavelet Transform
        wavelet_dwt_action = QAction('Discrete Wavelet Transform', wavelet_menu)
        wavelet_dwt_action.setStatusTip(
            'Use discrete wavelet transform to decompose/reconstruct as filtering or get rid of noise')
        wavelet_dwt_action.triggered.connect(self.waveletDWTDialog)
        wavelet_menu.addAction(wavelet_dwt_action)

        # Filter-Wavelet-Denoise
        wavelet_threshold_action = QAction('Denoise', wavelet_menu)
        wavelet_threshold_action.setStatusTip('Denoise input data depending on the mode argument')
        wavelet_threshold_action.triggered.connect(self.waveletThresholdDialog)
        wavelet_menu.addAction(wavelet_threshold_action)

        wavelet_menu.addSeparator()

        # Filter-Wavelet-Wavelet Packets
        wavelet_packet_action = QAction('Wavelet Packets', wavelet_menu)
        wavelet_packet_action.setStatusTip(
            'Use Wavelet Packets to decompose data into subnodes and reconstruct from ones needed')
        wavelet_packet_action.triggered.connect(self.waveletPacketsDialog)
        wavelet_menu.addAction(wavelet_packet_action)

    def initImage(self):
        """初始化绘图区"""

        plt.rc('font', family='Times New Roman')
        # plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        pg.setConfigOptions(leftButtonPan=True)  # 设置可用鼠标缩放
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')  # 设置界面前背景色

        # 绘制通道-时间图
        self.plot_channels_time_widget = MyPlotWidget('MultiChannel Cloud Image', 'Time(s)', 'Channel')
        self.plot_channels_time_widget.setXRange(0, 5)
        self.plot_channels_time_widget.setYRange(0, 200)  # 设置xy轴默认范围

        # 绘制灰度图
        self.plot_gray_scale_image = MyPlotWidget('Gray Scale Image', 'Time(s)', 'Channel')
        self.plot_gray_scale_image.setXRange(0, 5)
        self.plot_gray_scale_image.setYRange(0, 200)

        # 绘制单通道相位差-时间图
        self.plot_single_channel_time_widget = MyPlotWidget('Phase Difference Image', 'Time(s)',
                                                            'Phase Difference(rad)',
                                                            grid=True)
        self.plot_single_channel_time_widget.setXRange(0, 5)

        # 绘制频谱图
        self.plot_amplitude_frequency_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)',
                                                            'Amplitude', grid=True)
        self.plot_amplitude_frequency_widget.setXRange(0, 500)

        combine_image_widget = QWidget()
        image_vbox = QVBoxLayout()
        image_vbox.addWidget(self.plot_single_channel_time_widget)
        image_vbox.addWidget(self.plot_amplitude_frequency_widget)
        combine_image_widget.setLayout(image_vbox)

        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)
        self.tab_widget.setStyleSheet('font-size: 15px; font-family: "Times New Roman";')
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested[int].connect(self.removeTab)
        self.tab_widget.addTab(self.plot_channels_time_widget, 'Channels - Time')
        self.tab_widget.addTab(self.plot_gray_scale_image, 'Gray Scale')
        self.tab_widget.addTab(combine_image_widget, 'Single')
        self.tab_widget.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tab_widget.tabBar().setTabButton(1, QTabBar.RightSide, None)
        self.tab_widget.tabBar().setTabButton(2, QTabBar.RightSide, None)  # 设置删除按钮消失

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

        getPicture(file_path_jpg, 'file_path.jpg')
        change_file_path_button = PushButton('')
        change_file_path_button.setIcon(QIcon('file_path.jpg'))
        os.remove('file_path.jpg')
        change_file_path_button.setStyleSheet('background-color: rgb(255, 255, 255)')
        change_file_path_button.clicked.connect(self.changeFilePath)

        file_table_scrollbar = QScrollBar(Qt.Vertical)
        file_table_scrollbar.setStyleSheet('min-height: 100')
        self.files_table_widget = QTableWidget(100, 1)
        self.files_table_widget.setVerticalScrollBar(file_table_scrollbar)
        self.files_table_widget.setStyleSheet('font-size: 17px; font-family: "Times New Roman";')
        self.files_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.files_table_widget.setHorizontalHeaderLabels(['File'])
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
        sampling_rate_lable = Label('Sampling Rate')
        self.sampling_rate_line_edit = OnlyNumLineEdit()
        self.sampling_rate_line_edit.setFocusPolicy(Qt.NoFocus)

        sampling_times_lable = Label('Sampling Times')
        self.current_sampling_times_line_edit = OnlyNumLineEdit()
        self.current_sampling_times_line_edit.setFocusPolicy(Qt.NoFocus)

        number_of_channels_label = Label('Number of Channels')
        self.current_channels_line_edit = OnlyNumLineEdit()
        self.current_channels_line_edit.setFocusPolicy(Qt.NoFocus)

        channel_number_label = Label('Channel')
        self.channel_number_line_edit = OnlyNumLineEdit()

        self.channel_number_line_edit.textChanged.connect(self.changeChannelNumber)
        self.channel_number_line_edit.textChanged.connect(self.plotSingleChannelTime)
        self.channel_number_line_edit.textChanged.connect(self.plotAmplitudeFrequency)

        # 数据参数布局
        data_params_hbox = QHBoxLayout()
        data_params_hbox.addWidget(channel_number_label)
        data_params_hbox.addWidget(self.channel_number_line_edit)
        data_params_hbox.addSpacing(5)
        data_params_hbox.addWidget(sampling_rate_lable)
        data_params_hbox.addWidget(self.sampling_rate_line_edit)
        data_params_hbox.addSpacing(5)
        data_params_hbox.addWidget(sampling_times_lable)
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
        main_window_hbox.setStretchFactor(main_window_vbox, 4)
        main_window_widget.setLayout(main_window_hbox)
        self.setCentralWidget(main_window_widget)

    def initSetting(self):
        """初始化读取文件数"""

        self.files_read_number = 1

    def initParams(self):
        """初始化默认参数"""

        # 输出设置
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

        # 默认错误为None
        self.err = None

        # 通道数
        self.channel_from_num = 0
        self.channel_number = 0
        self.channel_to_num = self.current_channels

        # 采样数
        self.sampling_times_from_num = 0
        self.sampling_times_to_num = self.sampling_times
        self.current_sampling_times = self.sampling_times_to_num - self.sampling_times_from_num

        # 数据单位
        self.data_units = ['phase difference', 'strain rate']
        self.data_unit_index = 0
        self.strain_rate_action.setChecked(False)
        self.phase_difference_action.setChecked(True)

        # 二值图
        self.binary_image_flag = True
        self.binary_image_threshold = 120.0
        self.binary_image_threshold_methods = ['Two Peaks', 'OSTU']
        self.binary_image_threshold_method_index = 0

        # 加窗
        self.window_length = 32
        self.window_text = 'Rectangular / Dirichlet'
        self.window_method = boxcar
        self.window_overlap_size_ratio = 0.5
        self.window_overlap_size = int(round(self.window_overlap_size_ratio * self.window_length))  # 取整
        self.window_methods = {'Bartlett': bartlett, 'Blackman': blackman, 'Blackman-Harris': blackmanharris,
                               'Bohman': bohman, 'Cosine': cosine, 'Flat Top': flattop,
                               'Hamming': hamming, 'Hann': hann, 'Lanczos / Sinc': lanczos,
                               'Modified Barrtlett-Hann': barthann, 'Nuttall': nuttall, 'Parzen': parzen,
                               'Rectangular / Dirichlet': boxcar, 'Taylor': taylor, 'Triangular': triang,
                               'Tukey / Tapered Cosine': tukey}

        # 滤波器是否更新数据
        self.if_update_data = False

        # EMD
        self.imf_nums = 5
        self.reconstruct_nums = str([i for i in range(1, 5)])
        self.emd_options_flag = True
        self.eemd_trials = 100
        self.eemd_noise_width = 0.05
        self.ceemdan_trials = 100
        self.ceemdan_epsilon = 0.005
        self.ceemdan_noise_scale = 1.0
        self.ceemdan_noise_kind_index = 0
        self.ceemdan_range_thr = 0.01
        self.ceemdan_total_power_thr = 0.05

        # 小波分解
        self.wavelet_dwt_flag = True
        self.wavelet_dwt_reconstruct = ['cA1', 'cD1']
        self.wavelet_dwt_family_index = 0
        self.wavelet_dwt_name_index = 0
        self.wavelet_dwt_decompose_level = 1
        self.wavelet_dwt_decompose_level_calculated = False
        self.wavelet_dwt_padding_mode_index = 0

        # 小波去噪
        self.wavelet_threshold = 1.0
        self.wavelet_threshold_sub = 0.0
        self.wavelet_threshold_modes = ['soft', 'hard', 'garrote', 'greater', 'less']
        self.wavelet_threshold_mode_index = 0

        # 小波包分解
        self.wavelet_packets_flag = True
        self.wavelet_packets_reconstruct = ['a', 'd']
        self.wavelet_packets_family_index = 0
        self.wavelet_packets_name_index = 0
        self.wavelet_packets_decompose_level = 1
        self.wavelet_packets_decompose_max_level = None
        self.wavelet_packets_padding_mode_index = 0

    # """------------------------------------------------------------------------------------------------------------"""
    """一些特殊函数"""

    def closeEvent(self, event):
        """退出时的提示"""

        reply = QMessageBox.question(self, 'Tip', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def removeTab(self, index):
        """关闭对应选项卡"""

        self.tab_widget.removeTab(index)

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制四个固有图的函数"""

    def plotChannelsTime(self):
        """绘制通道-时间图"""

        self.plot_channels_time_widget.clear()

        self.plot_channels_time_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                                 self.sampling_times_to_num / self.sampling_rate)
        self.plot_channels_time_widget.setYRange(0, self.current_channels)
        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        colors = ['red', 'lime', 'deepskyblue', 'yellow', 'plum', 'gold', 'blue', 'fuchsia', 'aqua', 'orange']
        for i in range(self.current_channels):
            self.plot_channels_time_widget.plot(x, self.data[i, :] + i,
                                                pen=QColor(colors[i - 10 * (i // 10)]))  # 根据通道数个位选择颜色绘图

    def plotSingleChannelTime(self):
        """绘制单通道相位差-时间图"""

        self.plot_single_channel_time_widget.clear()

        data = self.data[self.channel_number, :]

        self.plot_single_channel_time_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                                       self.sampling_times_to_num / self.sampling_rate)
        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.data_unit_index == 0:
            self.plot_single_channel_time_widget.setTitle(
                '<font face="Times New Roman" size="5">Phase Difference Image</font>')
            self.plot_single_channel_time_widget.setLabel('left',
                                                          '<font face="Times New Roman">Phase Difference(rad)</font>')
        else:
            self.plot_single_channel_time_widget.setTitle(
                '<font face="Times New Roman" size="5">Strain Rate Image</font>')
            self.plot_single_channel_time_widget.setLabel('left',
                                                          '<font face="Times New Roman">Strain Rate(s^-1)</font>')

        self.plot_single_channel_time_widget.plot(x, data, pen=QColor('blue'))

    def plotAmplitudeFrequency(self):
        """绘制幅值-频率图"""

        self.plot_amplitude_frequency_widget.clear()

        data = self.data[self.channel_number, :]

        data = toAmplitude(data, self.current_sampling_times)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        self.plot_amplitude_frequency_widget.setXRange(0, self.sampling_rate / 2)
        y = fixDateLength(self.current_sampling_times)
        self.plot_amplitude_frequency_widget.plot(x, data[:y // 2], pen=QColor('blue'))

    def plotGrayChannelsTime(self):
        """绘制灰度 通道-时间图"""

        data = normalize(self.data)

        tr = QTransform()
        tr.scale(1 / self.sampling_rate, 1)
        tr.translate(self.sampling_times_from_num, 0)

        self.plot_gray_scale_image.clear()
        self.plot_gray_scale_image.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                             self.sampling_times_to_num / self.sampling_rate)
        self.plot_gray_scale_image.setYRange(0, self.current_channels)
        item = pg.ImageItem()
        item.setImage(data.T)
        item.setTransform(tr)
        self.plot_gray_scale_image.addItem(item)

    # """------------------------------------------------------------------------------------------------------------"""
    """文件路径区和文件列表调用函数"""

    def changeFilePath(self):
        """更改显示的文件路径"""

        self.file_path = QFileDialog.getExistingDirectory(self, "Select File Path", "C:/")  # 起始路径
        if self.file_path != '':
            self.updateFile()
        else:
            del self.file_path

    def selectDataFromTable(self):
        """当从文件列表中选择文件时更新图像等"""

        self.file_name = []
        item_index = self.files_table_widget.currentIndex().row()
        for i in range(self.files_read_number):
            if item_index + i + 1 > self.files_table_widget.rowCount():
                break
            self.file_name.append(self.files_table_widget.item(item_index + i, 0).text())
        self.readData()
        self.initParams()
        self.update()

    def changeChannelNumber(self):
        """更改通道号"""

        if self.channel_number_line_edit.text() == '':
            channel_number = 0
        else:
            channel_number = int(self.channel_number_line_edit.text())

        if channel_number <= self.current_channels - 1:
            self.channel_number = channel_number
        else:
            self.channel_number = self.current_channels - 1

        self.channel_number_line_edit.setText(str(self.channel_number))

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
        self.files_table_widget.setRowCount(len(files))
        for i in range(len(files)):
            table_widget_item = QTableWidgetItem(files[i])
            self.files_table_widget.setItem(i, 0, table_widget_item)

    def updateDataParams(self):
        """更新数据相关参数"""

        self.current_channels = self.channel_to_num - self.channel_from_num
        self.current_sampling_times = self.sampling_times_to_num - self.sampling_times_from_num

        self.channel_number_line_edit.setText(str(self.channel_number))
        self.sampling_rate_line_edit.setText(str(self.sampling_rate))
        self.current_sampling_times_line_edit.setText(str(self.current_sampling_times))
        self.current_channels_line_edit.setText(str(self.current_channels))

    def updateDataGPSTime(self):
        """更新数据时间显示"""

        # 更新开头文件GPS时间
        year = str(self.time[0][0])[:-2]
        month = str(self.time[0][1])[:-2]
        day = str(self.time[0][2])[:-2]
        hour = str(self.time[0][3])[:-2]
        minute = str(self.time[0][4])[:-2]
        second = str(self.time[0][5])
        self.gps_from_line_edit.setText(
            year + ' - ' + month + ' - ' + day + ' - ' + hour + ' - ' + minute + ' - ' + second)

        # 更新末尾文件GPS时间
        year = str(self.time[-1][0])[:-2]
        month = str(self.time[-1][1])[:-2]
        day = str(self.time[-1][2])[:-2]
        hour = str(self.time[-1][3])[:-2]
        minute = str(self.time[-1][4])[:-2]
        second = str(self.time[-1][5])
        self.gps_to_line_edit.setText(
            year + ' - ' + month + ' - ' + day + ' - ' + hour + ' - ' + minute + ' - ' + second)

    def updateImages(self):
        """更新4个随时更新的图像显示"""

        self.plotChannelsTime()
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
        """导入多个数据文件后绘制各种图"""

        self.file_name = QFileDialog.getOpenFileNames(self, 'Import', '', 'DAS data (*.dat)')  # 打开多个.dat文件
        self.file_name = self.file_name[0]
        if self.file_name != []:
            self.file_path = os.path.dirname(str(self.file_name[0]))
            self.readData()
            self.initParams()
            self.update()

    def readData(self):
        """读取数据，更新参数"""

        if self.file_name:
            # list(self.file_name).sort()
            time, data = [], []
            for file in self.file_name:
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
            self.sampling_times = sampling_times * len(self.file_name)
            self.current_channels = channels_num
            self.origin_data = data
            self.data = self.origin_data
            self.time = time

    # """------------------------------------------------------------------------------------------------------------"""
    """File-Export调用函数"""

    def exportData(self):
        """导出数据"""

        # files = self.file_name
        # files.sort()
        # files = ', '.join(files)
        #
        # gps_time = np.zeros_like(self.time)
        # for i in range(np.shape(self.time)[0]):
        #     for j in range(np.shape(self.time)[1]):
        #         gps_time[i][j] = round(self.time[i][j])
        #
        # comment = {'Export Time': f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}',
        #            'Exporter': f'{getpass.getuser()}', 'Source Data Files': f'{files}',
        #            'Source Data GPS Time From': f'{int(gps_time[0][0])}-{int(gps_time[0][1])}-{int(gps_time[0][2])} '
        #                                         f'{int(gps_time[0][3])}:{int(gps_time[0][4])}:{int(gps_time[0][5])}',
        #            'Source Data GPS Time To': f'{int(gps_time[-1][0])}-{int(gps_time[-1][1])}-{int(gps_time[-1][2])} '
        #                                       f'{int(gps_time[-1][3])}:{int(gps_time[-1][4])}:{int(gps_time[-1][5])}',
        #            'Source Data Parameters': {'Channels': f'{self.origin_data.shape[0]}',
        #                                       'Sampling Rate': f'{self.sampling_rate}Hz',
        #                                       'Sampling Times': f'{self.origin_data.shape[1]}'},
        #            'Current Data Parameters': {'Channels': f'{self.data.shape[0]}',
        #                                        'Sampling Rate': f'{self.sampling_rate}Hz',
        #                                        'Sampling Times': f'{self.data.shape[1]}'},
        #            'Export Format': 'pandas.DataFrame with shape(channels, sampling times)'}

        fpath, ftype = QFileDialog.getSaveFileName(self, 'Export', '',
                                                   'csv(*.csv);;json(*.json);;pickle(*.pickle);;txt(*.txt);;xls(*.xls *.xlsx)')

        data = pd.DataFrame(self.data)

        if ftype.find('*.txt') > 0:
            data.to_csv(fpath, sep=' ', index=False, header=False)
        elif ftype.find('*.csv') > 0:
            data.to_csv(fpath, sep=',', index=False, header=False)
        elif ftype.find('*.xls') > 0:
            data.to_csv(fpath, sep='\t', index=False, header=False)
        elif ftype.find('*.json') > 0:
            data.to_json(fpath, orient='values')
        elif ftype.find('*.pickle') > 0:
            data.to_pickle(fpath)

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

        from_num = int(self.sampling_times_from.text())
        to_num = int(self.sampling_times_to.text())
        list_num = [from_num, to_num]

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

        try:
            self.data = self.data[:, self.sampling_times_from_num:self.sampling_times_to_num]
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

        from_num = int(self.channel_from.text())
        to_num = int(self.channel_to.text())
        list_num = [from_num, to_num]

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

        try:
            self.data = self.data[self.channel_from_num:self.channel_to_num, :]
        except Exception as err:
            printError(err)

    # """------------------------------------------------------------------------------------------------------------"""
    """转换数据单位调用的函数"""

    def convert2PhaseDifference(self):
        """还原为相位差"""

        self.data_unit_index = 0
        self.data = self.origin_data[self.channel_from_num:self.channel_to_num,
                    self.sampling_times_from_num:self.sampling_times_to_num]
        self.updateImages()
        self.phase_difference_action.setChecked(True)
        self.strain_rate_action.setChecked(False)

    def convert2StrainRate(self):
        """相位差转换为应变率"""

        self.data_unit_index = 1
        self.data = phaseDifferenceToStrainRate(self.data, self.sampling_rate)
        self.updateImages()
        self.strain_rate_action.setChecked(True)
        self.phase_difference_action.setChecked(False)

    # """------------------------------------------------------------------------------------------------------------"""
    """更改读取文件数调用的函数"""

    def filesReadNumberDialog(self):
        """从表格选择文件时读取的文件数"""

        dialog = QDialog()
        dialog.setWindowTitle('Files Read Number')

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
    """计算数据特征调用的函数"""

    def plotTimeDomainFeature(self):
        """绘制选中的时域特征图像"""

        features = calculateTimeDomainFeatures(self.data)
        self.plotFeature(self.time_domain_chars_text, features)

    def plotFrequencyDomainFeature(self):
        """绘制选中的频域特征图像"""

        features = calculateFrequencyDomainFeatures(self.data, self.sampling_rate)
        self.plotFeature(self.fre_domain_chars_text, features)

    def plotFeature(self, text_list, features):
        """获取要计算的数据特征名字和值"""

        feature_name = self.plot_data_features_menu.sender().text()
        for key, value in text_list.items():
            if value == feature_name:
                feature_text = key
        index = [index for index, value in enumerate(text_list) if value == feature_text]
        feature_value = features[index[0]]
        plot_widget = MyPlotWidget(feature_name, 'Channel', '')

        x = np.linspace(0, self.current_channels, self.current_channels)
        plot_widget.setXRange(0, self.current_channels)
        plot_widget.plot(x, feature_value, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'{feature_name} - Window Method={self.window_text}')

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制应变图调用函数"""

    def plotStrain(self):
        """将相位差转为应变率再积分"""

        data = phaseDifferenceToStrain(self.data, self.sampling_rate)[self.channel_number, :]
        data *= 10e6

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        plot_widget = MyPlotWidget('Strain Image', 'Time(s)', 'Strain(με)', grid=True)
        plot_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        plot_widget.plot(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Strain Image - Channel Number={self.channel_number}')

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
        self.binary_image_method_combx.addItems(self.binary_image_threshold_methods)
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
            if self.binary_image_threshold_method_index == 0:
                threshold = twoPeaks(self.data)
            else:
                threshold = OSTU(self.data)

        self.binary_image_threshold_method_index = self.binary_image_method_combx.currentIndex()
        self.binary_image_threshold = threshold

    def plotBinaryImage(self):
        """绘制二值图"""

        binary_image_widget = MyPlotWidget('Binary Image', 'Sampling Times', 'Channel')
        self.tab_widget.addTab(binary_image_widget,
                               f'Binary Image - Threshold={self.binary_image_threshold}')  # 添加二值图tab

        # 阈值化
        binary_data = normalize(self.data)
        binary_data[binary_data >= self.binary_image_threshold] = 255
        binary_data[binary_data < self.binary_image_threshold] = 0  # 根据阈值赋值

        binary_image_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                      self.sampling_times_to_num / self.sampling_rate)
        binary_image_widget.setYRange(0, self.current_channels)

        tr = QTransform()
        tr.scale(1 / self.sampling_rate, 1)
        tr.translate(self.sampling_times_from_num, 0)

        binary_image = pg.ImageItem(binary_data.T)
        binary_image.setTransform(tr)
        binary_image_widget.addItem(binary_image)

    # """------------------------------------------------------------------------------------------------------------"""
    """绘制PSD调用函数"""

    def plotPSD(self):
        """绘制psd图线"""

        data = self.data[self.channel_number, :]
        data = self.window_method(self.current_sampling_times) * data
        data = np.abs(np.fft.fft(data))
        y = fixDateLength(self.current_sampling_times)
        data = 20.0 * np.log10(data ** 2 / self.current_sampling_times)[:y // 2]

        plot_widget = MyPlotWidget('PSD', 'Frequency(Hz)', 'Power/Frequency(dB/Hz)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.setXRange(0, self.sampling_rate / 2)
        plot_widget.setLogMode(x=True)
        plot_widget.plot(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'PSD - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dPSD(self):
        """绘制2dpsd谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget,
                               f'2D PSD - Window Method={self.window_text}\n'
                               f'Channel Number={self.channel_number}')
        data = self.data[self.channel_number, :]
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

        data = self.data[self.channel_number, :]
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

        data = self.data[self.channel_number, :]
        y = fixDateLength(self.current_sampling_times)
        data = self.window_method(self.current_sampling_times) * data
        data = 20.0 * np.log10(np.abs(np.fft.fft(data)) / self.current_sampling_times)[:y // 2]

        plot_widget = MyPlotWidget('Magnitude Spectrum', 'Frequency(Hz)', 'Magnitude(dB)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.plot(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Magnitude Spectrum - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dMagnitudeSpectrum(self):
        """绘制2d幅度谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'2D Magnitude Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        data = self.data[self.channel_number, :]
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

        data = self.data[self.channel_number, :]
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

        data = self.data[self.channel_number, :]
        y = fixDateLength(self.current_sampling_times)
        data = self.window_method(self.current_sampling_times) * data
        data = np.angle(np.fft.fft(data))[:y // 2]

        plot_widget = MyPlotWidget('Angle Spectrum', 'Frequency(Hz)', 'Angle(rad)', grid=True)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        plot_widget.plot(x, data, pen=QColor('blue'))
        self.tab_widget.addTab(plot_widget, f'Angle Spectrum - Window Method={self.window_text}\n'
                                            f'Channel Number={self.channel_number}')

    def plot2dAngleSpectrum(self):
        """绘制2d相位谱"""

        figure = plt.figure()
        widget = FigureCanvas(figure)
        self.tab_widget.addTab(widget, f'2D Angle Spectrum - Window Method={self.window_text}\n'
                                       f'Channel Number={self.channel_number}')

        data = self.data[self.channel_number, :]
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

        data = self.data[self.channel_number, :]
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

        btn = PushButton('OK')
        btn.clicked.connect(self.updateWindowOptionsParams)
        btn.clicked.connect(self.plotSingleChannelTime)
        btn.clicked.connect(self.plotAmplitudeFrequency)
        btn.clicked.connect(dialog.close)

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
            self.window_overlap_size_ratio = 0.99
        else:
            self.window_overlap_size_ratio = float(self.window_overlap_size_ratio_line_edit.text())
        self.window_overlap_size = int(round(self.window_overlap_size_ratio * self.window_length))

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter更新数据调用函数"""

    def updateFilteredData(self):
        """判断是否更新数据"""

        if self.update_data_action.text() == 'Update Data(False)':
            self.if_update_data = True
            self.update_data_action.setText('Update Data(True)')
        else:
            self.if_update_data = False
            self.update_data_action.setText('Update Data(False)')

    # """------------------------------------------------------------------------------------------------------------"""
    """Filter-EMD调用函数"""

    def plotEMD(self):
        """绘制emd分解图和重构图"""

        self.emd_method = self.emd_menu.sender().text()
        data = self.data[self.channel_number, :]

        if self.data_unit_index == 1:
            data *= 10e6
        try:
            if self.emd_method == 'EMD':
                emd = EMD()
                self.imfs_res = emd.emd(data, max_imf=self.imf_nums - 1)
            elif self.emd_method == 'EEMD':
                emd = EEMD(trials=self.eemd_trials, noise_width=self.eemd_noise_width)
                self.imfs_res = emd.eemd(data, max_imf=self.imf_nums - 1)
            elif self.emd_method == 'CEEMDAN':
                if not hasattr(self, 'ceemdan_noise_kind_combx'):
                    noise_kind = 'normal'
                else:
                    noise_kind = self.ceemdan_noise_kind_combx.currentText()
                emd = CEEMDAN(trials=self.ceemdan_trials, epsilon=self.ceemdan_epsilon,
                              noise_scale=self.ceemdan_noise_scale,
                              noise_kind=noise_kind, range_thr=self.ceemdan_range_thr,
                              total_power_thr=self.ceemdan_total_power_thr)
                self.imfs_res = emd.ceemdan(data, max_imf=self.imf_nums - 1)
        except Exception as err:
            printError(err)

        if self.data_unit_index == 1:
            self.imfs_res /= 10e6

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.emd_options_flag:
            wgt = QWidget()
            wgt.setFixedWidth(self.tab_widget.width())
            vbox1 = QVBoxLayout()
            vbox2 = QVBoxLayout()
            hbox = QHBoxLayout()
            scroll_area = QScrollArea()

            pw_time_list = []
            pw_fre_list = []
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
                pw_time.plot(x_time, self.imfs_res[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)

                data = toAmplitude(self.imfs_res[i], self.current_sampling_times)
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(self.current_sampling_times)
                pw_fre.plot(x_fre, data[:y // 2], pen=QColor('blue'))
                pw_fre_list.append(pw_fre)
                pw_fre_list[i].setXLink(pw_fre_list[0])

                vbox1.addWidget(pw_time)
                vbox2.addWidget(pw_fre)

            hbox.addLayout(vbox1)
            hbox.addLayout(vbox2)
            wgt.setLayout(hbox)
            scroll_area.setWidget(wgt)
            self.tab_widget.addTab(scroll_area,
                                   f'{self.emd_method} - Decompose: Number of IMF={self.imf_nums - 1}\n'
                                   f'Channel Number={self.channel_number}')
        else:
            reconstruct_imf = [int(i) for i in re.findall('\d+', self.reconstruct_nums)]
            data = np.zeros(self.imfs_res[0].shape)
            for i in range(len(reconstruct_imf) - 1):
                imf_num = reconstruct_imf[i]
                data += self.imfs_res[imf_num, :]

            if self.data_unit_index == 0:
                data_widget = MyPlotWidget(f'{self.emd_method} Reconstruct', 'Times', 'Phase Difference(rad)',
                                           grid=True)
            else:
                data_widget = MyPlotWidget(f'{self.emd_method} Reconstruct', 'Times', 'Strain Rate(s^-1)', grid=True)

            data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  self.sampling_times_to_num / self.sampling_rate)
            data_widget.plot(x, data, pen=QColor('blue'))

            data = toAmplitude(data, self.current_sampling_times)
            x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
            fre_amp_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)', 'Amplitude', grid=True)
            fre_amp_widget.setXRange(0, self.sampling_rate / 2)
            y = fixDateLength(self.current_sampling_times)
            fre_amp_widget.plot(x, data[:y // 2], pen=QColor('blue'))

            vbox = QVBoxLayout()
            combine_widget = QWidget()
            vbox.addWidget(data_widget)
            vbox.addWidget(fre_amp_widget)
            combine_widget.setLayout(vbox)
            self.tab_widget.addTab(combine_widget, f'Reconstruct: Number of IMF={reconstruct_imf}')

            if self.if_update_data:
                self.data[self.channel_number, :] = data
                self.updateImages()

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
        self.emd_decompose_line_edit.setText(str(self.imf_nums - 1))

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
                self.imf_nums = 2
            elif int(self.emd_decompose_line_edit.text()) >= 10:
                self.imf_nums = 10
            else:
                self.imf_nums = int(self.emd_decompose_line_edit.text()) + 1
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
            pw_list[i].plot(x, inst_freqs[i, :], pen=QColor('blue'))
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

        self.filter = FilterI('Bessel/Thomson')

        self.filter.btn.clicked.connect(self.plotIIRFilter)
        self.filter.btn.clicked.connect(self.filter.dialog.close)
        self.filter.dialog.exec_()

    def iirDesignCombFilter(self):
        """设计comb类滤波器"""

        self.filter = FilterII(self.iir_menu.sender().text())

        self.filter.btn.clicked.connect(self.plotIIRFilter)
        self.filter.btn.clicked.connect(self.filter.dialog.close)
        self.filter.dialog.exec_()

    def plotIIRFilter(self):
        """绘制iir滤波器图"""

        data = self.data[self.channel_number, :]
        data = filtfilt(self.filter.b, self.filter.a, data)

        if self.if_update_data:
            self.data[self.channel_number, :] = data
            self.updateImages()

        combine_widget = QWidget()
        vbox = QVBoxLayout()

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.data_unit_index == 0:
            data_widget = MyPlotWidget('Phase Difference Image', 'Time(s)', 'Phase Difference(rad)', grid=True)
        else:
            data_widget = MyPlotWidget('Strain Rate Image', 'Time(s)', 'Strain Rate(s^-1)', grid=True)

        data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        data_widget.plot(x, data, pen=QColor('blue'))

        data = toAmplitude(data, self.current_sampling_times)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        fre_amp_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)', 'Amplitude', grid=True)
        fre_amp_widget.setXRange(0, self.sampling_rate / 2)
        y = fixDateLength(self.current_sampling_times)
        fre_amp_widget.plot(x, data[:y // 2], pen=QColor('blue'))

        vbox.addWidget(data_widget)
        vbox.addWidget(fre_amp_widget)
        combine_widget.setLayout(vbox)
        if hasattr(self.filter, 'method'):
            self.tab_widget.addTab(combine_widget,
                                   f'Filtered Image - Filter={self.filter.name}\n'
                                   f'Method={self.filter.method}\t'
                                   f'Channel Number={self.channel_number}')
        else:
            self.tab_widget.addTab(combine_widget,
                                   f'Filtered Image - Filter={self.filter.name}\n'
                                   f'Channel Number={self.channel_number}')

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

        data = self.data[self.channel_number, :]
        if self.data_unit_index == 1:
            data *= 10e6

        if self.wavelet_dwt_decompose_level_calculated:
            self.wavelet_dwt_decompose_level = pywt.dwt_max_level(self.current_sampling_times,
                                                                  self.wavelet_dwt_name_combx.currentText())

        if self.wavelet_dwt_flag:
            try:
                self.wavelet_dwt_coeffs = pywt.wavedec(data, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                                       mode=self.wavelet_dwt_padding_mode_combx.currentText(),
                                                       level=self.wavelet_dwt_decompose_level)
            except Exception as err:
                printError(err)

            self.wavelet_dwt_reconstruct = []
            self.wavelet_dwt_reconstruct.append(f'cA{self.wavelet_dwt_decompose_level}')
            for i in range(len(self.wavelet_dwt_coeffs) - 1, 0, -1):
                self.wavelet_dwt_reconstruct.append(f'cD{i}')

        else:
            rec_coeffs = self.wavelet_dwt_reconstruct.split("'")
            rec_Dcoeffs = []
            for i in rec_coeffs:
                if re.match('cA\d+|cD\d+', i) is None:
                    continue
                rec_Dcoeffs.append(re.match('cA\d+|cD\d+', i).group())

            rec_Dcoeffs_number = []
            for i in rec_Dcoeffs[1:]:
                rec_Dcoeffs_number.append(int(re.match('\d+', i[2:]).group()))

            for i in rec_Dcoeffs_number:
                if i not in range(len(self.wavelet_dwt_coeffs) - 1):
                    self.wavelet_dwt_coeffs[-i] = np.zeros_like(self.wavelet_dwt_coeffs[-i])

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

            pw_time_list = []
            pw_fre_list = []
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
                pw_time.plot(x_time, coeffs[i], pen=QColor('blue'))
                pw_time_list.append(pw_time)
                # pw_time_list[i].setXLink(pw_time_list[0])

                data = toAmplitude(coeffs[i], len(coeffs[i]))
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / len(coeffs[i]))
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(len(coeffs[i]))
                pw_fre.plot(x_fre, data[:y // 2], pen=QColor('blue'))
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
            if self.data_unit_index == 0:
                data_widget = MyPlotWidget('DWT Reconstruct', 'Time(s)', 'Phase Difference(rad)', grid=True)
            else:
                data_widget = MyPlotWidget('DWT Reconstruct', 'Time(s)', 'Strain Rate(s^-1)', grid=True)

            x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                            self.current_sampling_times) / self.sampling_rate

            try:
                data = pywt.waverec(coeffs, wavelet=self.wavelet_dwt_name_combx.currentText(),
                                    mode=self.wavelet_dwt_padding_mode_combx.currentText())
            except Exception as err:
                printError(err)

            data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  self.sampling_times_to_num / self.sampling_rate)
            data_widget.plot(x, data, pen=QColor('blue'))

            data = toAmplitude(data, self.current_sampling_times)
            x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
            fre_amp_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)', 'Amplitude', grid=True)
            fre_amp_widget.setXRange(0, self.sampling_rate / 2)
            y = fixDateLength(self.current_sampling_times)
            fre_amp_widget.plot(x, data[:y // 2], pen=QColor('blue'))

            vbox = QVBoxLayout()
            combine_widget = QWidget()
            vbox.addWidget(data_widget)
            vbox.addWidget(fre_amp_widget)
            combine_widget.setLayout(vbox)
            self.tab_widget.addTab(combine_widget,
                                   f'DWT - Reconstruct: Coefficient={self.wavelet_dwt_reconstruct}\n'
                                   f'Wavelet={self.wavelet_dwt_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')

            if self.if_update_data:
                self.data[self.channel_number, :] = data
                self.updateImages()

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

        data = self.data[self.channel_number, :]

        try:
            data = pywt.threshold(data, value=self.wavelet_threshold,
                                  mode=self.wavelet_threshold_mode_combx.currentText(),
                                  substitute=self.wavelet_threshold_sub)
        except Exception as err:
            printError(err)

        if self.if_update_data:
            self.data[self.channel_number, :] = data
            self.updateImages()

        combine_widget = QWidget()
        vbox = QVBoxLayout()

        x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                        self.current_sampling_times) / self.sampling_rate
        if self.data_unit_index == 0:
            data_widget = MyPlotWidget('Phase Difference Image', 'Time(s)', 'Phase Difference(rad)', grid=True)
        else:
            data_widget = MyPlotWidget('Strain Rate Image', 'Time(s)', 'Strain Rate(s^-1)', grid=True)

        data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                              self.sampling_times_to_num / self.sampling_rate)
        data_widget.plot(x, data, pen=QColor('blue'))

        data = toAmplitude(data, self.current_sampling_times)
        x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
        fre_amp_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)', 'Amplitude', grid=True)
        fre_amp_widget.setXRange(0, self.sampling_rate / 2)
        y = fixDateLength(self.current_sampling_times)
        fre_amp_widget.plot(x, data[:y // 2], pen=QColor('blue'))

        vbox.addWidget(data_widget)
        vbox.addWidget(fre_amp_widget)
        combine_widget.setLayout(vbox)

        self.tab_widget.addTab(combine_widget,
                               f'Wavelet Thresholded Image - Threshold={self.wavelet_threshold}\n'
                               f'Threshold Type={self.wavelet_threshold_mode_combx.currentText()}\t'
                               f'Channel Number={self.channel_number}')

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
            self.wavelet_packets_decompose_max_level = pywt.dwt_max_level(self.data.shape[1], wavelet)
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

        data = self.data[self.channel_number, :]
        if self.data_unit_index == 1:
            data *= 10e6

        try:
            if self.wavelet_packets_flag:
                self.wavelet_packets_wp = pywt.WaveletPacket(data,
                                                             wavelet=self.wavelet_packets_name_combx.currentText(),
                                                             mode=self.wavelet_packets_padding_mode_combx.currentText())
                self.wavelet_packets_subnodes = self.wavelet_packets_wp.get_level(
                    level=self.wavelet_packets_decompose_level, order='natural', decompose=True)
                self.wavelet_packets_reconstruct = [i.path for i in self.wavelet_packets_subnodes]

            else:
                total_paths = [i.path for i in self.wavelet_packets_subnodes]
                self.wavelet_packets_reconstruct = self.wavelet_packets_reconstruct.split("','")
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

            pw_time_list = []
            pw_fre_list = []
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
                pw_time.plot(x_time, subnodes[i].data, pen=QColor('blue'))
                pw_time_list.append(pw_time)

                data = toAmplitude(subnodes[i].data, len(subnodes[i].data))
                x_fre = np.arange(0, self.sampling_rate / 2, self.sampling_rate / len(subnodes[i].data))
                pw_fre.setXRange(0, self.sampling_rate / 2)
                pw_fre.setFixedHeight(150)
                y = fixDateLength(len(subnodes[i].data))
                pw_fre.plot(x_fre, data[:y // 2], pen=QColor('blue'))
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
            if self.data_unit_index == 0:
                data_widget = MyPlotWidget('Wavelet Packets Reconstruct', 'Time(s)', 'Phase Difference(rad)', grid=True)
            else:
                data_widget = MyPlotWidget('Wavelet Packets Reconstruct', 'Time(s)', 'Strain Rate(s^-1)', grid=True)

            try:
                data = self.wavelet_packets_wp.reconstruct()
            except Exception as err:
                printError(err)

            x = np.linspace(self.sampling_times_from_num, self.sampling_times_to_num,
                            self.current_sampling_times) / self.sampling_rate
            data_widget.setXRange(self.sampling_times_from_num / self.sampling_rate,
                                  self.sampling_times_to_num / self.sampling_rate)
            data_widget.plot(x, data, pen=QColor('blue'))

            data = toAmplitude(data, self.current_sampling_times)
            x = np.arange(0, self.sampling_rate / 2, self.sampling_rate / self.current_sampling_times)
            fre_amp_widget = MyPlotWidget('Amplitude - Frequency Image', 'Frequency(Hz)', 'Amplitude', grid=True)
            fre_amp_widget.setXRange(0, self.sampling_rate / 2)
            y = fixDateLength(self.current_sampling_times)
            fre_amp_widget.plot(x, data[:y // 2], pen=QColor('blue'))

            vbox = QVBoxLayout()
            combine_widget = QWidget()
            vbox.addWidget(data_widget)
            vbox.addWidget(fre_amp_widget)
            combine_widget.setLayout(vbox)
            self.tab_widget.addTab(combine_widget,
                                   f'Wavelet Packets - Reconstruct: Subnodes={self.wavelet_packets_reconstruct}\n'
                                   f'Wavelet={self.wavelet_packets_name_combx.currentText()}\t'
                                   f'Channel Number={self.channel_number}')

            if self.if_update_data:
                self.data[self.channel_number, :] = data
            self.updateImages()

    # """------------------------------------------------------------------------------------------------------------"""


if __name__ == '__main__':
    DAS_Visualizer = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(DAS_Visualizer.exec_())
