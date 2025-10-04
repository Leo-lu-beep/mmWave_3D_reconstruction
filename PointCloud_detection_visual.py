# -*- coding: gbk -*-
import numpy as np
import matplotlib.pyplot as plt

class PointCloud_detection_visual:
    def __init__(self):
        super().__init__()
    
    def range_heatmap(self, rd_matrix, param, frameIdx, rxIdx, logScale=True):
        # ѡ���ض����ߺ�֡������
        data = rd_matrix[rxIdx, :, :, frameIdx]
        # �Ƴ���ά�ȣ�ѹ��Ϊ��ά����
        data = np.squeeze(data)
        # ���������
        powerSpectrum = np.abs(data) ** 2
        # ת��ΪdB scale
        if logScale:
            powerSpectrum = 10 * np.log10(powerSpectrum + 1e-10)
        
        # ����ͼ��
        plt.figure(figsize=(8, 6))
         
        # ����������
        range_bins = np.arange(1, param.rangeFFTSize + 1)
        range_axis = range_bins * param.rangeRes
        chirp_axis = np.arange(1, param.numChirpsPerTX + 1)
        # ������ͼ
        im = plt.imshow(powerSpectrum.T, aspect='auto', 
                        extent=[range_axis[0], range_axis[-1], chirp_axis[0], chirp_axis[-1]],
                        origin='lower')
        plt.set_cmap('jet')
        cbar = plt.colorbar(im)
        if logScale:
            cbar.set_label('Power Spectrum (dB)')
        else:
            cbar.set_label('Power Spectrum')
        
        # ���ñ���ͱ�ǩ
        plt.title(f'RangeFFT Heatmap - RX_idx: {rxIdx}, Frame_idx: {frameIdx}')
        plt.xlabel('Range(m)')
        plt.ylabel('Chirp Index')
        
        # �������
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # ��ʾͼ��
        plt.tight_layout()
        plt.show()
