# -*- coding: utf-8 -*-
"""
@Time    : 2024/9/30 下午0:25
@Author  : zxy
@File    : binary_image.py
"""
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout

from utils.widget import PushButton, ComboBox, RadioButton, LineEditWithReg, Dialog


class BinaryImage:
    binary_image_threshold_methods = {'双峰法': 'twoPeaks', '大津法': 'ostu'}

    def __init__(self):
        self.data = None
        self.binary_image_flag = True  # 是否使用简单阈值
        self.binary_image_threshold = 120.0  # 阈值
        self.binary_image_threshold_method_index = 0  # 计算阈值方法的索引

        self.draw = False

    def normalizeToGrayScale(self, data: np.array) -> np.array:
        """数据标准化至0-255"""
        mx, mn = np.max(data), np.min(data)
        data = (data - mn) / (mx - mn)
        return data * 255

    def binarizeData(self) -> None:
        """二值化数据"""
        self.draw = True
        self.data[self.data >= self.binary_image_threshold] = 255
        self.data[self.data < self.binary_image_threshold] = 0

    def runDialog(self) -> None:
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
        btn.clicked.connect(self.binarizeData)
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

    def updateBinaryImageParams(self) -> None:
        """检查选择那种求阈值方法"""
        self.binary_image_flag = self.binary_image_input_radiobtn.isChecked()

        if self.binary_image_flag:
            threshold = float(self.binary_image_threshold_line_edit.text())
            if threshold > 255:
                threshold = 255
        else:
            threshold = getattr(self,
                                self.binary_image_threshold_methods[self.binary_image_method_combx.currentText()])()

        self.binary_image_threshold_method_index = self.binary_image_method_combx.currentIndex()
        self.binary_image_threshold = threshold

    def twoPeaks(self) -> int:
        """双峰法求阈值"""

        # 存储灰度直方图
        gray_scale_hist = [0] * 256
        for i, x in enumerate(self.data):
            for j, y in enumerate(x):
                gray_scale_hist[int(y)] += 1

        # 寻找灰度直方图的最大峰值对应的灰度值
        first_peak = gray_scale_hist.index(max(gray_scale_hist))

        # 寻找灰度直方图的第二个峰值对应的灰度值
        distance = [(i - first_peak) ** 2 * gray_scale_hist[i] for i in range(256)]  # 综合考虑 两峰距离与峰值
        second_peak = distance.index(max(distance))

        # 找到两个峰值之间的最小值对应的灰度值，作为阈值
        if first_peak > second_peak:
            first_peak, second_peak = second_peak, first_peak

        temp_val = gray_scale_hist[first_peak:second_peak]
        min_gray_scale_location = temp_val.index(min(temp_val))
        return first_peak + min_gray_scale_location + 1

    def ostu(self) -> int:
        """OSTU（大津）法"""
        height, width = self.data.shape
        sq = height * width
        max_gray_scale = threshold = 0
        # 遍历每一个灰度层
        for i in range(256):
            smaller_px = self.data[np.where(self.data < i)]
            bigger_px = self.data[np.where(self.data >= i)]
            smaller_ratio = len(smaller_px) / sq
            bigger_ratio = len(bigger_px) / sq
            average_gray_scale_smaller = np.mean(smaller_px) if len(smaller_px) > 0 else 0
            average_gray_scale_bigger = np.mean(bigger_px) if len(bigger_px) > 0 else 0
            otsu = smaller_ratio * bigger_ratio * (average_gray_scale_smaller - average_gray_scale_bigger) ** 2
            if otsu > max_gray_scale:
                max_gray_scale = otsu
                threshold = i
        return threshold

    def run(self, data: np.array) -> Optional[int]:
        """求二值图阈值"""
        self.draw = False
        self.data = self.normalizeToGrayScale(data)
        self.runDialog()
        return self.data if self.draw else None
