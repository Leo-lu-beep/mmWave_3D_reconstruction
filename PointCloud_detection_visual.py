# -*- coding: gbk -*-
import numpy as np
import matplotlib.pyplot as plt

class PointCloud_detection_visual:
    def __init__(self):
        super().__init__()
    
    def range_heatmap(self, rd_matrix, param, frameIdx, rxIdx, logScale=True):
        # 选择特定天线和帧的数据
        data = rd_matrix[rxIdx, :, :, frameIdx]
        # 移除单维度，压缩为二维矩阵
        data = np.squeeze(data)
        # 计算幅度谱
        powerSpectrum = np.abs(data) ** 2
        # 转换为dB scale
        if logScale:
            powerSpectrum = 10 * np.log10(powerSpectrum + 1e-10)
        
        # 创建图像
        plt.figure(figsize=(8, 6))
         
        # 设置坐标轴
        range_bins = np.arange(1, param.rangeFFTSize + 1)
        range_axis = range_bins * param.rangeRes
        chirp_axis = np.arange(1, param.numChirpsPerTX + 1)
        # 绘制热图
        im = plt.imshow(powerSpectrum.T, aspect='auto', 
                        extent=[range_axis[0], range_axis[-1], chirp_axis[0], chirp_axis[-1]],
                        origin='lower')
        plt.set_cmap('jet')
        cbar = plt.colorbar(im)
        if logScale:
            cbar.set_label('Power Spectrum (dB)')
        else:
            cbar.set_label('Power Spectrum')
        
        # 设置标题和标签
        plt.title(f'RangeFFT Heatmap - RX_idx: {rxIdx}, Frame_idx: {frameIdx}')
        plt.xlabel('Range(m)')
        plt.ylabel('Chirp Index')
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图像
        plt.tight_layout()
        plt.show()
