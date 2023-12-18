"""数据计算、处理及输出错误，每次打包之前运行一下"""

import base64
import os

import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QAction, QMenu
from scipy import stats, integrate


def picture2py(pic_name):
    """将图像文件转换为py文件"""

    write_data = []
    for picture_name in os.listdir('Image/'):
        filename = picture_name.replace('.', '_')
        open_pic = open(os.path.join('Image/', picture_name), 'rb')
        b64str = base64.b64encode(open_pic.read())
        open_pic.close()
        # 注意这边b64str一定要加上.decode()
        write_data.append('{} = "{}"\n'.format(filename, b64str.decode()))

    f = open('{}.py'.format(pic_name), 'w+')
    for data in write_data:
        f.write(data)
    f.close()


def getPicture(pic_code, pic_name):
    """创建图片"""

    image = open(pic_name, 'wb')
    image.write(base64.b64decode(pic_code))
    image.close()


def setPicture(picture, picture_name, widget):
    """设置图片"""

    getPicture(picture, picture_name)  # 从image.py中获取图片信息生成图片
    widget.setIcon(QIcon(picture_name))  # 加载图片
    os.remove(picture_name)  # 移除图片释放内存
    widget.setStyleSheet('background: white')


if __name__ == '__main__':
    picture2py('image')


def printError(err):
    """将捕获的错误输出"""

    msgbx = QMessageBox.warning(None, '错误', f'<font face="Times New Roman" size="4">{err}</font>!',
                                QMessageBox.Ok)

    return


def createMenu(text, parent, status_tip=None, enabled=True):
    """创建菜单"""

    menu = QMenu(text, parent)
    menu.setStatusTip(status_tip)
    parent.addMenu(menu)
    menu.setEnabled(enabled)

    return menu


def createAction(text, parent, status_tip, connect_func, short_cut=0):
    """创建动作菜单"""

    action = QAction(text, parent)
    action.setStatusTip(status_tip)
    action.setShortcut(short_cut)
    action.triggered.connect(connect_func)
    parent.addAction(action)

    return action


def normalizeToGrayScale(data):
    """数据标准化至0-255"""

    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data *= 255

    return normalized_data


def toAmplitude(data, sampling_times):
    """fft后数据处理作幅值图"""

    data = np.abs(np.fft.fft(data)) / sampling_times
    data[0] = 0
    data *= 2

    return data


def calculateTimeDomainFeatures(data):
    """计算时域特征"""

    rows, cols = data.shape

    # 有量纲统计量
    max_value = np.amax(data, axis=1)  # 最大值
    peak_value = np.amax(np.abs(data), axis=1)  # 最大绝对值
    min_value = np.amin(data, axis=1)  # 最小值
    mean = np.mean(data, axis=1)  # 均值
    peak_peak_value = max_value - min_value  # 峰峰值
    mean_absolute_value = np.mean(np.abs(data), axis=1)  # 绝对平均值
    root_mean_square = np.sqrt(np.sum(data ** 2, axis=1) / cols)  # 均方根值
    square_root_amplitude = (np.sum(np.sqrt(np.abs(data)), axis=1) / cols) ** 2  # 方根幅值
    variance = np.var(data, axis=1)  # 方差
    standard_deviation = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=1, fisher=False)  # 峭度
    skewness = stats.skew(data, axis=1)  # 偏度
    # 无量纲统计量
    clearance_factor = peak_value / square_root_amplitude  # 裕度指标
    shape_factor = root_mean_square / mean_absolute_value  # 波形指标
    impulse_factor = peak_value / mean_absolute_value  # 脉冲指标
    crest_factor = peak_value / root_mean_square  # 峰值指标
    kurtosis_factor = kurtosis / (root_mean_square ** 4)  # 峭度指标

    features = {'最大值': max_value, '峰值': peak_value, '最小值': min_value, '平均值': mean, '峰峰值': peak_peak_value,
                '绝对平均值': mean_absolute_value, '均方根值': root_mean_square, '方根幅值': square_root_amplitude,
                '方差': variance, '标准差': standard_deviation, '峭度': kurtosis, '偏度': skewness,
                '裕度因子': clearance_factor, '波形因子': shape_factor, '脉冲因子': impulse_factor, '峰值因子': crest_factor,
                '峭度因子': kurtosis_factor}

    return features


def calculateFrequencyDomainFeatures(data, sampling_rate):
    """计算频域特征"""

    data_fft = np.fft.fft(data, axis=1)
    channel_num, data_length = data_fft.shape  # 样本个数和信号长度

    # 傅里叶变换是对称的，只需取前半部分数据，否则由于频率序列是正负对称的，会导致计算重心频率求和时正负抵消
    magnitude = np.abs(data_fft)[:, :data_length // 2]  # 信号幅值
    frequency = np.fft.fftfreq(data_length, 1 / sampling_rate)[:data_length // 2]

    power_spectrum = magnitude ** 2 / data_length  # 功率谱

    centroid_frequency = np.sum(frequency * power_spectrum, axis=1) / np.sum(power_spectrum, axis=1)  # 重心频率
    mean_frequency = np.mean(power_spectrum, axis=1)  # 平均频率
    mean_square_frequency = np.sum(power_spectrum * np.square(frequency), axis=1) / np.sum(power_spectrum,
                                                                                           axis=1)  # 均方频率
    root_mean_square_frequency = np.sqrt(mean_square_frequency)  # 均方根频率

    freq_tile = np.tile(frequency.reshape(1, -1), (channel_num, 1))  # 复制 channel_num 行
    fc_tile = np.tile(centroid_frequency.reshape(-1, 1), (1, freq_tile.shape[1]))  # 复制列，与freq_tile的列数对应
    frequency_variance = np.sum(np.square(freq_tile - fc_tile) * power_spectrum, axis=1) / np.sum(power_spectrum,
                                                                                                  axis=1)  # 频率方差
    frequency_standard_deviation = np.sqrt(frequency_variance)  # 频率标准差

    features = {'重心频率': centroid_frequency, '平均频率': mean_frequency, '均方根频率': root_mean_square_frequency,
                '均方频率': mean_square_frequency, '频率方差': frequency_variance, '频率标准差': frequency_standard_deviation}

    return features


def twoPeaks(data):
    """双峰法求阈值"""

    data = normalizeToGrayScale(data)
    rows, cols = data.shape

    # 存储灰度直方图
    gray_scale_hist = np.zeros([256], np.uint64)
    for i in range(rows):
        for j in range(cols):
            gray_scale_hist[round(data[i][j])] += 1

    # 寻找灰度直方图的最大峰值对应的灰度值
    max_gray_scale_location = np.where(gray_scale_hist == np.max(gray_scale_hist))
    first_peak = max_gray_scale_location[0][0]  # 灰度值
    # 寻找灰度直方图的第二个峰值对应的灰度值
    distance = np.zeros([256], np.float32)
    for i in range(256):
        distance[i] = pow(i - first_peak, 2) * gray_scale_hist[i]  # 综合考虑 两峰距离与峰值
    max_gray_scale_location2 = np.where(distance == np.max(distance))
    second_peak = max_gray_scale_location2[0][0]

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    if first_peak > second_peak:  # 第一个峰值再第二个峰值的右侧
        temp_val = gray_scale_hist[int(second_peak):int(first_peak)]
        min_gray_scale_location = np.where(temp_val == np.min(temp_val))
        threshold = second_peak + min_gray_scale_location[0][0] + 1
    else:  # 第一个峰值再第二个峰值的左侧
        temp_val = gray_scale_hist[int(first_peak):int(second_peak)]
        min_gray_scale_location = np.where(temp_val == np.min(temp_val))
        threshold = first_peak + min_gray_scale_location[0][0] + 1

    return threshold


def OSTU(data):
    """OSTU（大津）法"""

    data = normalizeToGrayScale(data)
    height, width = data.shape
    max_gray_scale = 0
    threshold = 0
    # 遍历每一个灰度层
    for i in range(255):
        # 使用numpy直接对数组进行计算
        smaller_px = data[np.where(data < i)]
        bigger_px = data[np.where(data >= i)]
        smaller_ratio = len(smaller_px) / (height * width)
        bigger_ratio = len(bigger_px) / (height * width)
        average_gray_scale_smaller = np.mean(smaller_px) if len(smaller_px) > 0 else 0
        average_gray_scale_bigger = np.mean(bigger_px) if len(bigger_px) > 0 else 0
        otsu = smaller_ratio * bigger_ratio * (average_gray_scale_smaller - average_gray_scale_bigger) ** 2
        if otsu > max_gray_scale:
            max_gray_scale = otsu
            threshold = i

    return threshold


def convertDataUnit(data, sr, src, aim):
    """数据单位转换"""

    if src == 'PD':
        if aim == 'SR':
            return data * 11.6e-9 * sr
        elif aim == 'S':
            data = data * 11.6e-9 * sr
            x = data.shape[1]
            x = np.linspace(0, x, x)
            return integrate.cumtrapz(data, x, initial=0) * 1e6
    elif src == 'SR':
        if aim == 'PD':
            return data / 11.6e-9 / sr
        elif aim == 'S':
            x = data.shape[1]
            x = np.linspace(0, x, x)
            return integrate.cumtrapz(data, x, initial=0) * 1e6
