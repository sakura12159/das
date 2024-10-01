# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/12 上午9:27
@Author  : zxy
@File    : function.py
"""
import base64
import os
from typing import Dict, Union

import numpy as np
import soundfile
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QWidget
from scipy import stats
from scipy.signal import detrend


def printError(err: Union[Exception, str]) -> None:
    """将捕获的错误输出"""
    QMessageBox.warning(None, '错误', f'<font face="Times New Roman" size="4">{err}</font>!', QMessageBox.Ok)


def writeImages(path: str) -> None:
    """初始化所需图片"""
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
    """创建图片"""
    image = open(picture_name, 'wb')
    image.write(base64.b64decode(picture_code))
    image.close()


def setPicture(widget: QWidget, picture_code: str, picture_name: str, window_icon: bool = False) -> None:
    """设置图片"""
    getPicture(picture_code, picture_name)  # 从image.py中获取图片信息生成图片
    if window_icon:
        widget.setWindowIcon(QIcon(picture_name))
    else:
        widget.setIcon(QIcon(picture_name))  # 加载图片
        widget.setStyleSheet('background: white')
    os.remove(picture_name)  # 移除图片释放内存


def writeWav(path: str, data: np.array, sr: int) -> None:
    """创建wav文件"""
    soundfile.write(os.path.join(path, 'temp.wav'), data, sr)
