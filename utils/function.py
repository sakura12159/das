# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/12 上午9:27
@Author  : zxy
@File    : function.py
"""
import base64
import os
from typing import Union, Optional

import numpy as np
import soundfile
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout
from scipy.signal import detrend

from utils.widget import MyPlotWidget


def printError(err: Union[Exception, str]) -> None:
    """
    将捕获的错误输出
    Args:
        err: 捕获的错误

    Returns:

    """
    QMessageBox.warning(None, '错误', f'<font face="Times New Roman" size="4">{err}</font>!', QMessageBox.Ok)


def writeImages(path: str) -> None:
    """
    初始化所需图片
    Args:
        path: 图片保存路径

    Returns:

    """
    pic_data = []
    picture_names = [i for i in os.listdir(path) if i.endswith('.jpg')]
    for picture_name in picture_names:
        filename = picture_name.replace('.', '_')
        open_pic = open(os.path.join(path, picture_name), 'rb')
        b64str = base64.b64encode(open_pic.read())
        open_pic.close()
        pic_data.append(f'{filename} = "{b64str.decode()}"\n')

    f = open(os.path.join(path, 'image.py'), 'w+')
    f.writelines(pic_data)
    f.close()


def getPicture(picture_code: str, picture_name: str) -> None:
    """
    创建图片
    Args:
        picture_code: 文件流
        picture_name: 文件名

    Returns:

    """
    image = open(picture_name, 'wb')
    image.write(base64.b64decode(picture_code))
    image.close()


def setPicture(widget: QWidget, picture_code: str, picture_name: str, window_icon: bool = False) -> None:
    """
    设置图片
    Args:
        widget: 要设置图片的组件
        picture_code: 文件流
        picture_name: 文件名
        window_icon: 设置的是否为主窗口图标

    Returns:

    """
    getPicture(picture_code, picture_name)  # 从image.py中获取图片信息生成图片
    if window_icon:
        widget.setWindowIcon(QIcon(picture_name))
    else:
        widget.setIcon(QIcon(picture_name))  # 加载图片
        widget.setStyleSheet('background: white')
    os.remove(picture_name)  # 移除图片释放内存


def writeWav(path: str, data: np.array, sr: int) -> None:
    """
    创建 temp.wav 文件
    Args:
        path: 创建路径
        data: 要写入的数据
        sr: 数据的采样率

    Returns:

    """
    soundfile.write(os.path.join(path, 'temp.wav'), data, sr)


def detrendData(data: np.array) -> np.array:
    """
    去除信号的均值和线性趋势
    Args:
        data: 数据

    Returns: 去趋势后的数据

    """
    data = detrend(data, axis=1, type='constant')
    data = detrend(data, axis=1, type='linear')
    return data


def toAmplitude(data: np.array) -> np.array:
    """
    一维时域数据转幅值
    Args:
        data: 数据

    Returns: 数据幅值

    """
    data = np.abs(np.fft.fft(data)) * 2.0 / len(data)
    return data[:len(data) // 2]


def xAxis(num: int,
          begin: Optional[int] = None,
          end: Optional[int] = None,
          sampling_rate: int = 1,
          freq: bool = False) -> np.array:
    """
    生成绘制时域图的 x 轴
    Args:
        num: 数据点数量
        begin: 数据轴开始值
        end: 数据轴结束值
        sampling_rate: 采样率
        freq: 是否为频域 x 轴

    Returns: 生成的 x 轴

    """
    if freq:
        return np.fft.fftfreq(num, 1 / sampling_rate)[:num // 2]
    return np.linspace(begin, end, num) / sampling_rate


def initCombinedPlotWidget(data: np.array,
                           title: str,
                           begin: int,
                           end: int,
                           num: int,
                           sampling_rate: int,
                           ) -> QWidget:
    """
    创建返回结合两个 pw 的 Qwidget
    Args:
        data: 数据
        title: 组件名
        begin: 数据轴开始值
        end: 数据轴结束值
        num: 数据点数量
        sampling_rate: 采样率

    Returns: 生成的组合组件

    """
    x = xAxis(num, begin, end, sampling_rate)
    data_widget = MyPlotWidget(f'{title}', '时间（s）', '相位差（rad）', grid=True)
    data_widget.draw(x, data, pen=QColor('blue'))

    data = toAmplitude(data)
    x = xAxis(num, sampling_rate=sampling_rate, freq=True)
    fre_amp_widget = MyPlotWidget('幅值图', '频率（Hz）', '幅值', grid=True)
    fre_amp_widget.draw(x, data, pen=QColor('blue'))

    combined_widget = QWidget()
    vbox = QVBoxLayout()
    vbox.addWidget(data_widget)
    vbox.addWidget(fre_amp_widget)
    combined_widget.setLayout(vbox)
    return combined_widget
