# -*- coding: gbk -*-
import os
import numpy as np
from scipy.fft import  fft, fftshift
import matplotlib.pyplot as plt
from PointCloud_detection_utils import PointCloud_detection_utils
class RadarParameters:
    # mmWave Radar config parameters
    def __init__(self):
        self.numFrames = 300            # ��֡��
        self.numADCSamples = 128        # ADC��������
        self.numChirps = 129            # ��chirp��
        self.numADCBits = 16            # ADCλ��
        self.numTX = 3                  # ����������
        self.numRX = 4                  # ����������
        self.numAnt = self.numTX * self.numRX
        self.ADCValidStartTime = 6e-6   # ADC��Ч������ʼʱ�� us
        self.RampEndTime = 65e-6        # ����chirp�źŵ��ܳ���ʱ�䣨��Ƶ�ʿ�ʼ�仯��������ʱ�䣻ʱ��Խ�����״��̽�����Խ����Ƶ��б�ʹ�ͬ������Ч����Ӱ�����ֱ��ʣ� us
        self.IdleTime = 7e-6            # ����chirp�ź�֮��ļ��ʱ�䣬�����״���Ƶǰ�˵ġ���λ������������chirp�źŵĸ��ţ�ͬʱΪ���ݴ�����������ռ�
        self.isReal = False             # trueΪʵ���� FalseΪ������
        self.Fs = 4.4e6                 # ADCÿ��ɼ�������������λ��Hz�������ο�˹�ز��������������������ź����Ƶ�ʵ�2��
        self.c = 3e8                    # ���� m/s
        self.slope = 60.012e12
        self.startFreq = 77e9           # ��ʼƵ��
        self.FramePeriodicity = 100e-3  # ֡���� ms
        
        # ������������
        self.Tc = self.numADCSamples / self.Fs  # ADC����ʱ��
        self.bandWidthValid = self.Tc * self.slope  # ��Чɨ�����
        self.lambda_ = self.c / self.startFreq  # ����
        self.Tr = self.numTX * (self.IdleTime + self.RampEndTime)  # ���� chirp �źŵ��ܳ���ʱ��
        self.d = self.lambda_ / 2  # ���߼��
        self.numChirpsPerTX = int(np.floor(self.numChirps / self.numTX)) # ÿ��TX��chirp��
        
        # FFT��������
        self.rangeFFTSize = self.numADCSamples  # ��ΧFFT��С
        self.dopplerFFTSize = self.numChirpsPerTX  # ������FFT��С
        self.angleFFTSize = 180  # �Ƕ�FFT��С
        self.angleGridDeg = np.linspace(-90, 90, 1801)
        
        # ���ܲ���
        self.maxRange = 8.792
        self.maxDoppler = 4.435
        self.rangeRes = self.c / (2 * self.bandWidthValid)
        self.dopplerRes = 0.069
        
        self.scale = np.arange(-60, 60.2, 0.2)
        self.slowFs = 1 / self.Tr
        self.indexSpace = np.linspace(-self.numChirpsPerTX / 2, self.numChirpsPerTX / 2, self.numChirpsPerTX)
        self.dopplerIndex = self.indexSpace * (self.slowFs * self.lambda_) / (self.numChirpsPerTX*2)
    
    def printParams(self):
        # ��ӡ�����״����
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}")
            
class CFAR_Parameters:
    def __init__(self,radarParams):
        self.pfa = 10e-3
        self.rangeFFTSize = radarParams.rangeFFTSize
        self.dopplerFFTSzie = radarParams.dopplerFFTSize
        self.rangeRes = radarParams.rangeRes
        self.dopplerRes = radarParams.dopplerRes
    def printParams(self):
        # ��ӡ�����״����
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}")

class RD_Grid_Params:
    def __init__(self,radarParams):
        self.rangeInterval = radarParams.rangeRes
        self.dopplerInterval = radarParams.dopplerRes
        self.color = 'r' # ��ɫ������
        self.lineStyle = '-.' # �㻮����ʽ
        self.lineWidth = 1 # �߿�Ϊ1
    def printParams(self):
        # ��ӡ�����״����
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}")

class PointCloud_detection_visual:
    def __init__(self):
        super().__init__()
    
    def range_heatmap(self, rdMatrix, param, frameIdx, rxIdx, logScale=True):
        # ѡ���ض����ߺ�֡������
        data = rdMatrix[rxIdx, :, :, frameIdx]
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
    def doppler_heatmap(self, rdMatrix, param, frameIdx, rxIdx, logScale=True, gridParams=None, title1=None, title2=None):
        # ��ȡָ�����ߺ�֡������
        data = np.squeeze(rdMatrix[rxIdx, :, :, frameIdx])
        # ���㹦����
        powerSpectrum = np.abs(data)**2
        # �Ƿ���dB��ͼ
        if logScale:
            powerSpectrum = 10 * np.log10(powerSpectrum + 1e-10)
        # ����������
        rangeBins = np.arange(param.rangeFFTSize)
        rangeAxis = rangeBins * param.rangeRes
        dopplerBins = np.arange(param.dopplerFFTSize)
        dopplerAxis = (dopplerBins - param.dopplerFFTSize/2) * param.dopplerRes
        # ��������ͼ
        plt.figure()
        # ʹ��imshow��������ͼ��ע��ת��ƥ��MATLAB��imagesc��Ϊ
        im = plt.imshow(powerSpectrum.T, aspect='auto', 
                        extent=[rangeAxis[0], rangeAxis[-1], dopplerAxis[0], dopplerAxis[-1]],
                        origin='lower')
        plt.xlabel('Range(m)')
        plt.ylabel('doppler(m/s)')
        
        cbar = plt.colorbar(im)
        if logScale:
            cbar.set_label('Power Spectrum (dB)')
        else:
            cbar.set_label('Power Spectrum')
        
            # ����ṩ����������������������
        if gridParams is not None:
            # ���ƾ���������
            max_range = np.max(rangeAxis)
            range_ticks = np.arange(0, max_range + gridParams.rangeInterval, gridParams.rangeInterval)
            for r in range_ticks:
                plt.plot([r, r], [np.min(dopplerAxis), np.max(dopplerAxis)],
                        color=gridParams.color,
                        linestyle=gridParams.lineStyle,
                        linewidth=gridParams.lineWidth)
            
            # �����ٶ�������
            min_doppler = np.min(dopplerAxis)
            max_doppler = np.max(dopplerAxis)
            doppler_ticks = np.arange(min_doppler, max_doppler + gridParams.dopplerInterval, gridParams.dopplerInterval)
            for v in doppler_ticks:
                plt.plot([np.min(rangeAxis), np.max(rangeAxis)], [v, v],
                        color=gridParams.color,
                        linestyle=gridParams.lineStyle,
                        linewidth=gridParams.lineWidth)
        plt.title(f'{title1}: {rxIdx}, {title2}: {frameIdx}')
        plt.tight_layout()
        plt.show()

class doaParams():
    def __init__(self,radarParams):
        self.A_n = 8
        self.Fs = radarParams.Fs
        self.Tr = radarParams.Tr
        self.c = radarParams.c
        self.slope = radarParams.slope
        self.lambda_ = radarParams.lambda_
        self.d = radarParams.d # �������м��
        self.numADCSamples = radarParams.numADCSamples
        self.numChirpsPerTX = radarParams.numChirpsPerTX
        self.angleFFTSize = radarParams.angleFFTSize
        self.rangeFFTSize = radarParams.rangeFFTSize
        self.dopplerFFTSize = radarParams.dopplerFFTSize
    def printParams(self):
        # ��ӡ�����״����
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}") 

def topN_rd(dopplerCoherentPerFrame, nPeaks=10):
    """����RDͼ��N����ǿ���ȵ����ص�, ��������Ͷ�Ӧ�ĸ���ֵ

    Args:
        dopplerCoherentPerFrame (_type_): ����ĸ������󣬴���ÿ֡�Ķ������������
        nPeaks (int, optional): Ҫ��������ǿ���ص������. Defaults to 10.
    """
    RDM_abs = np.abs(dopplerCoherentPerFrame)  # magnitude map
    RDM_norm = RDM_abs / np.max(RDM_abs)  # normalize

    # Pre-allocate outputs
    RD_top10_mask = np.zeros_like(RDM_norm, dtype=bool)  # boolean mask
    RD_top10_complex = np.zeros_like(dopplerCoherentPerFrame, dtype=complex)  # complex output

    # Find indices of top N_peaks largest values
    # Flatten the array and get indices of largest values
    flat_indices = np.argpartition(RDM_norm.flatten(), -nPeaks)[-nPeaks:]
    
    # Convert flat indices to 2D indices
    rowIdx, colIdx = np.unravel_index(flat_indices, RDM_norm.shape)
    
    # Set mask and complex values at the top positions
    for k in range(len(flat_indices)):
        RD_top10_mask[rowIdx[k], colIdx[k]] = True
        RD_top10_complex[rowIdx[k], colIdx[k]] = dopplerCoherentPerFrame[rowIdx[k], colIdx[k]]
    
    return RD_top10_mask, RD_top10_complex

def generatePointCloud(dopplerOutResultPerFrame, rdTopNMask, doa_params):
    """���ݶ�����������ݺ�RDͼ��ֵ�������ɵ���(���롢�ٶȡ���λ�ǡ�������)

    Args:
        dopplerOutResultPerFrame (_type_): dopplerFFT��ĸ�������
        rdTopNMask (_type_): 2D bool ����(rangeFFTSize, dopplerFFTSize), RDͼ��ǰN����ֵ������
        doaParams (_type_): �����ֵ�
    Returns:
        results (_type_): 2D����(numTargets x 4), ÿ�д洢һ��Ŀ������������
                        [range(m), velocity(m/s), azimuth(rad), elevation(rad)]
    """
    # -------------------------- 1. ��ʼ��������м����� --------------------------
    # ���ƽ����ʼ��(������,������̬���Ŀ��)
    Results = np.zeros((0, 4), dtype=np.float64)
    Results_n = 0
    # ����������(2ͨ��x����binx������bin)
    elevationArrays = np.zeros((2, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # ��λ������(numAntͨ��x����binx������bin)
    azimuthArrays = np.zeros((doa_params.A_n, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # -------------------------- 2. ��丩����/��λ��ԭʼ���� --------------------------
    # Ϊ���������鸳ֵ �ؼ���MATLAB 1-based���� �� Python 0-based������5��4��3��2��
    elevationArrays[0, :, :] = dopplerOutResultPerFrame[4, :, :]
    elevationArrays[1, :, :] = dopplerOutResultPerFrame[2, :, :]
    
    # Ϊ��λ�����鸳ֵ�������Σ�1-4ͨ����5-8ͨ������ӦԭMATLAB�߼���
    for vn in range(4):  # ԭVn=1~4 �� Python vn=0~3
        azimuthArrays[vn, :, :] = dopplerOutResultPerFrame[vn, :, :]
    for vn in range(4,8): # ԭVn=5~8 �� Python vn=4~7����ӦԭVn+4��5+4=9��Python 8��
        azimuthArrays[vn, :, :] = dopplerOutResultPerFrame[vn + 4, :, :]
    # -------------------------- 3. ɸѡĿ��㣨��Ŀ����������㣩 --------------------------
    # ��ȡ��Ŀ��� / Ŀ������������ 
    comR, comD = np.where(rdTopNMask == False)  # ��Ŀ���
    # print(f"nonTargetR.shape: {nonTargetR.shape}, nonTargetR: {nonTargetR}")
    r1, c1 = np.where(rdTopNMask == True)  # Ŀ���
    # print(f"targetR: {r1}, targetD: {c1}")
    # ���ɷ�Ŀ���ı�ƽ�����루ԭMATLAB sub2ind���ܣ�
    mask = np.zeros((doa_params.rangeFFTSize, doa_params.dopplerFFTSize),dtype=bool)
    mask[comR, comD] = True
    
    # ��Ŀ���������0(������)
    for vn in range(2):
        layer = np.squeeze(elevationArrays[vn,:,:].copy())
        layer[mask] = 0 
        elevationArrays[vn,:,:] = layer
    # ��Ŀ���������0(��λ��)
    for vn in range(doa_params.A_n):
        layer = np.squeeze(azimuthArrays[vn,:,:].copy())
        layer[mask] = 0
        azimuthArrays[vn,:,:] = layer
    # -------------------------- 4. �Ƕ�FFT���㣨��λ��+�����ǣ� --------------------------
    # ��ʼ���Ƕ�FFT����洢����
    # ��λ��FFT��
    sigAfft = np.zeros((doa_params.angleFFTSize, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # ������FFT��
    sigEfft = np.zeros((doa_params.angleFFTSize, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # �洢ÿ��Ŀ���ĽǶ�FFT��ֵ����
    angleA = np.zeros(len(r1), dtype=np.int64)  # ��λ�Ƿ�ֵ����
    angleE = np.zeros(len(r1), dtype=np.int64)  # �����Ƿ�ֵ����
    index = 0 # ��������
    # ���Ŀ������Ƕ�FFT
    for i in range(len(c1)):
        row = r1[i]  # ��ǰĿ��ľ���bin��0-based��
        col = c1[i]  # ��ǰĿ��Ķ�����bin��0-based��
                       # ���Ŀ�����ȡ����

        # ��λ��FFT������+Ƶ����λ��
        tempA = azimuthArrays[:, row, col]  # ��ȡĿ���8������ͨ���ĸ�������
        temp_fftA = fftshift(fft(tempA, doa_params.angleFFTSize))
        sigAfft[:, row, col] = temp_fftA
        maxAIdx = np.argmax(np.abs(temp_fftA))  # Ѱ�ҽǶ�FFT��������ֵ�������
        angleA[index] = maxAIdx # �洢������ֵ�������
        
        # ������FFT������+Ƶ����λ��
        tempE = elevationArrays[:, row, col]  # ��ȡ�õ�2��������ͨ������
        tempE_fftE = fftshift(fft(tempE, n=doa_params.angleFFTSize))
        sigEfft[:, row, col] = tempE_fftE
        maxEIdx = np.argmax(np.abs(tempE_fftE))
        angleE[index] = maxEIdx # �洢������ֵ�������
        
        index += 1
    
    # -------------------------- 5. ����Ŀ��������� --------------------------
    # ������롢��λ�ǡ�������
    R_idx = -1  # ��ʼ��Ϊ-1����Ϊ������0��ʼ
    Azimuth = 0
    Elevation = 0
    Elevation_Phi = np.zeros((2, 0))  # ��ʼ��Ϊ������
    Elevation_Phi_diff = 0

     # ��ʼ��Results���飬Ԥ����ռ�
    num_targets = len(r1)
    Results = np.zeros((num_targets, 4))
    Results_n = 0
    
    for i in range(len(r1)):
        # �����һ���Ѿ�����һ�δ�����������������ظ�
        if r1[i] == R_idx:
            continue

        A_idx = angleA[i]  # ��Ŀ���ڷ�λ�����ϵķ�ֵBin
        E_idx = angleE[i]  # ��Ŀ���ڸ��������ϵķ�ֵBin
        R_idx = r1[i]  # Range_bin �������ھ���������
        C_idx = c1[i]  # Doppler_bin �������ڶ�����������

        # ��չElevation_Phi������������Ŀ��
        if Results_n >= Elevation_Phi.shape[1]:
            # ��̬��չ����
            new_cols = np.zeros((2, Elevation_Phi.shape[1] + 10))
            if Elevation_Phi.shape[1] > 0:
                new_cols[:, :Elevation_Phi.shape[1]] = Elevation_Phi
            Elevation_Phi = new_cols

        # ������λ��
        for Rn in range(2):
            Elevation_Phi[Rn, Results_n] = np.angle(elevationArrays[Rn, R_idx, C_idx])

        Elevation_Phi_diff = Elevation_Phi[0, Results_n] - Elevation_Phi[1, Results_n]
        Elevation_Phi_diff = np.mod(Elevation_Phi_diff + np.pi, 2 * np.pi) - np.pi
        
        # ���չ�ʽ������ز���
        Fb = ((R_idx) * doa_params.Fs) / doa_params.numADCSamples  # ע�⣺Python������0��ʼ
        Fd = (C_idx - doa_params.numChirpsPerTX / 2) / (doa_params.numChirpsPerTX * doa_params.Tr)  # ������Ƶ��
        Fa = (A_idx - doa_params.angleFFTSize / 2) / doa_params.angleFFTSize  # ��λ��Ƶ��
        Fe = (E_idx - doa_params.angleFFTSize / 2) / doa_params.angleFFTSize  # ������Ƶ��
    
        R = doa_params.c * Fb / (2 * doa_params.slope)  # �������
        V = doa_params.lambda_ * Fd / 2  # �ٶȽ��㣬ע�⣺lambda��Python�ؼ���
        Azimuth = np.arcsin(Fa * doa_params.lambda_ / doa_params.d)  # ��λ�ǶȽ���
        Elevation = np.arcsin(Elevation_Phi_diff * doa_params.lambda_ / (2 * np.pi * doa_params.d))
    
        # �洢���
        Results[Results_n, 0] = R
        Results[Results_n, 1] = V
        Results[Results_n, 2] = Azimuth
        Results[Results_n, 3] = Elevation
    
        Results_n += 1

    return Results

filePath = "/Volumes/ϵͳ/dataset/2025_07_22/adc_data.bin"

if __name__ == "__main__":
    radarParams = RadarParameters()
    cfarParams = CFAR_Parameters(radarParams)
    RdGridParams = RD_Grid_Params(radarParams)
    doa_params = doaParams(radarParams)
    pointCloudUtils = PointCloud_detection_utils()
    pointCloud_detection_visual = PointCloud_detection_visual()
    
    radarParams.printParams()
    cfarParams.printParams()
    RdGridParams.printParams()
    
    virtualAntennaArrayAfter = pointCloudUtils.bin_to_IQ(filePath,radarParams)
    print(virtualAntennaArrayAfter)
    
    # ��ʼ��ȫ0����
    rangeFFT_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames),dtype=complex)
    DopplerFFT_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames), dtype=complex)
    MTI_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames), dtype=complex)
    # ��֡����
    for nF in range(radarParams.numFrames):
        print(f"Processing frame {nF+1} of {radarParams.numFrames}")
        # ���ɺ�����
        rangeWindowCoeffVec = np.hanning(radarParams.rangeFFTSize)
        # ��ʼ���洢����
        rangeFFT_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.numChirpsPerTX), dtype=complex)
        # range FFT
        for i_an in range(radarParams.numAnt):
            # ��ȡ��Ӧά�ȵ����ݡ�squeeze�Ƴ���ά��
            RangeMatMeanBefore = np.squeeze(virtualAntennaArrayAfter[i_an,:,:,nF])
            # ȥ����ֵ����̬�Ӳ�ȥ����
            # MATLAB��meanĬ�ϰ��м��㣨ά��2����python����ָ��axis=0
            RangeMatMeanAfter = RangeMatMeanBefore - np.mean(RangeMatMeanBefore, axis=0, keepdims=True)
            # Ӧ�ô�����
            # repmat ��python�п���np.tile�����ע��ά��ƥ��
            # rangeWindowCoeffVec[:, np.newaxis]:��һά�Ĵ�����ϵ�������ӣ�N,��ת��Ϊ��N,1��np.tile():�����ظ����飻(1, radarParams.numChirpsPerTX)��1��ʾ�ڵ�һ��ά���ϲ��ظ����ڶ���������ʾ�ڵڶ���ά���ظ�ָ������
            # ���������������ϵ����N,1����չΪ��N,numChirpsPerTX����ά����
            RangeMat_window_after = RangeMatMeanAfter * np.tile(rangeWindowCoeffVec[:, np.newaxis], (1, radarParams.numChirpsPerTX))
            # ִ��FFT��ָ���ڵ�һ��ά���Ͻ��� (axis=0��ӦMATLAB��1)
            fftOutput_Range = np.fft.fft(RangeMat_window_after, radarParams.rangeFFTSize, axis=0)
            
            # �洢���
            rangeFFT_out_idx[i_an, :, :] = fftOutput_Range
            # print(f"fftOutput_Range.shape: {fftOutput_Range.shape}")
        # print(f"rangeFFT_out_idx.shape: {rangeFFT_out_idx.shape}")
        
        
        DopplerWindowCoeffVec = np.hanning(radarParams.dopplerFFTSize)
        # Ԥ����������飨�������ͣ�
        DopplerFFT_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize),dtype=complex)
        # doppler FFT
        for i_an in range(radarParams.numAnt):
            # ��ȡ��ά��Ƭ [rangeFFTSize x numChirpsPerTX]
            DopplerMatWinBefore = np.squeeze(rangeFFT_out_idx[i_an, :, :])
            DopplerMatWinAfter = DopplerMatWinBefore * np.tile(DopplerWindowCoeffVec[:, np.newaxis].T,(radarParams.rangeFFTSize,1))
            # ִ�ж�����FFT
            fftOutputDoppler = np.fft.fft(DopplerMatWinAfter, radarParams.dopplerFFTSize, axis=1)
            # print(fftOutput_Doppler.shape)
            # ����Ҷ��λ���ڶ���ά�ȣ�
            fftOutputDopplerShift = np.fft.fftshift(fftOutputDoppler, axes=1)
            # print("fftOutputDopplerShift.shape:",fftOutputDopplerShift.shape)
            # �洢���
            DopplerFFT_out_idx[i_an, :, :] = fftOutputDopplerShift
        
        # MTI
        if nF > 0:
            MTI_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize),dtype=complex)
            # ��ÿ�����뵥Ԫ�������յ�Ԫ������Ӧ��MTI�˲���
            for i_an in range(radarParams.numAnt):
                for i_range in range(radarParams.rangeFFTSize):
                    for i_doppler in range(radarParams.dopplerFFTSize):
                        # ��ȡ��ǰ֡��RD������
                        currentRdData = DopplerFFT_out_idx[i_an,i_range,i_doppler]
                        # ��ȡ��һ֡��ͬλ�õ�RD������
                        previousRdData = DopplerFFT_out_result[i_an,i_range,i_doppler,nF-1]
                        # Ӧ��MTI�˲���(һ�ζ�������ǰ֡-��һ֡)
                        MTI_out_idx[i_an,i_range,i_doppler] = currentRdData - previousRdData
            
            # ����MTI����������
            MTI_out_result[:, :, :, nF] = MTI_out_idx
        else:
            # ��һֱ֡�ӱ���ԭʼDopplerFFT���
            MTI_out_result[:, :, :, nF] = DopplerFFT_out_idx

        # �����м���
        rangeFFT_out_result[:, :, :, nF] = rangeFFT_out_idx
        DopplerFFT_out_result[:, :, :, nF] = DopplerFFT_out_idx

    # ѭ������
    pointCloud_detection_visual.range_heatmap(rdMatrix=rangeFFT_out_result, 
                                          param=radarParams, 
                                          frameIdx=0, 
                                          rxIdx=0, 
                                          logScale=False,)
    pointCloud_detection_visual.range_heatmap(rdMatrix=rangeFFT_out_result, 
                                            param=radarParams, 
                                            frameIdx=0, 
                                            rxIdx=0, 
                                            logScale=True)
    
    for i_nF in range(9,20):
        # ԭʼDoppler��ͼ(��һ������ͨ��)
        pointCloud_detection_visual.doppler_heatmap(rdMatrix=DopplerFFT_out_result, 
                                                    param=radarParams, 
                                                    frameIdx=i_nF, 
                                                    rxIdx=0, 
                                                    logScale=False, 
                                                    gridParams=None,
                                                    title1="Origin Doppler FFT Heatmap - RX",
                                                    title2="Frame")
        pointCloud_detection_visual.doppler_heatmap(rdMatrix=MTI_out_result, 
                                                    param=radarParams, 
                                                    frameIdx=i_nF, 
                                                    rxIdx=0, 
                                                    logScale=False, 
                                                    gridParams=None,
                                                    title1="MTI Processed Doppler FFT Heatmap - RX",
                                                    title2="Frame")
    
    # ������ά�Ƚ�����ɺϲ�������SNR
    DopplerCoherent = np.zeros((radarParams.rangeFFTSize, radarParams.numChirpsPerTX,radarParams.numFrames),dtype=complex)
    for i_nF in range(radarParams.numFrames):
        for i_channel in range(radarParams.numAnt):
            # �ۼ�MTI�����Ľ��
            DopplerCoherent[:, :, i_nF] += MTI_out_result[i_channel, :, :, i_nF]
    
    # ����֡10��20(MATLAB��10:20��Ӧpython��9:20)
    for i_nF in range(9,20):
        # ��ȡtop10����͸�������
        rdTopNMask, rdTopNComplex = topN_rd(DopplerCoherent[:, :, i_nF],nPeaks=10)
        Results = generatePointCloud(dopplerOutResultPerFrame=MTI_out_result[:, :, :, i_nF], 
                                rdTopNMask=rdTopNMask, 
                                doa_params=doa_params)
        print(Results.shape)
        # ��ȡʵ�ʽ��
        R = Results[:, 0]
        v = Results[:, 1]
        azimuth = np.rad2deg(Results[:,2])
        elevation = np.rad2deg(Results[:,3])
        # ���ӻ�����
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(azimuth, R, elevation, c=v, cmap='jet', s=20)
        plt.colorbar(scatter)
        
        ax.set_ylabel('Range (m)')
        ax.set_xlabel('Azimuth (��)')
        ax.set_zlabel('Elevation (��)')
        ax.set_title(f'Point-cloud: Range�CAz�CEl (colour = speed) - Frame {i_nF}')
        
        # ���������᷶Χ
        ax.set_ylim([0, (radarParams.rangeFFTSize - 1) * radarParams.rangeRes])
        ax.set_xlim([-90, 90])
        ax.set_zlim([-60, 60])
        
        plt.grid(True)
        plt.show()