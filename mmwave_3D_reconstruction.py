# -*- coding: gbk -*-
import os
import numpy as np
from scipy.fft import  fft, fftshift
import matplotlib.pyplot as plt
from PointCloud_detection_utils import PointCloud_detection_utils
class RadarParameters:
    # mmWave Radar config parameters
    def __init__(self):
        self.numFrames = 300            # 总帧数
        self.numADCSamples = 128        # ADC采样点数
        self.numChirps = 129            # 总chirp数
        self.numADCBits = 16            # ADC位宽
        self.numTX = 3                  # 发射天线数
        self.numRX = 4                  # 接收天线数
        self.numAnt = self.numTX * self.numRX
        self.ADCValidStartTime = 6e-6   # ADC有效数据起始时间 us
        self.RampEndTime = 65e-6        # 单个chirp信号的总持续时间（从频率开始变化到结束的时间；时间越长，雷达的探测距离越大；与频率斜率共同决定有效带宽，影响距离分辨率） us
        self.IdleTime = 7e-6            # 两个chirp信号之间的间隔时间，用于雷达射频前端的“复位”，避免相邻chirp信号的干扰，同时为数据处理留出缓冲空间
        self.isReal = False             # true为实采样 False为复采样
        self.Fs = 4.4e6                 # ADC每秒采集的样本数（单位：Hz）根据奈奎斯特采样定理，采样率至少是信号最高频率的2倍
        self.c = 3e8                    # 光速 m/s
        self.slope = 60.012e12
        self.startFreq = 77e9           # 起始频率
        self.FramePeriodicity = 100e-3  # 帧周期 ms
        
        # 计算派生参数
        self.Tc = self.numADCSamples / self.Fs  # ADC采样时间
        self.bandWidthValid = self.Tc * self.slope  # 有效扫描带宽
        self.lambda_ = self.c / self.startFreq  # 波长
        self.Tr = self.numTX * (self.IdleTime + self.RampEndTime)  # 单个 chirp 信号的总持续时间
        self.d = self.lambda_ / 2  # 天线间距
        self.numChirpsPerTX = int(np.floor(self.numChirps / self.numTX)) # 每个TX的chirp数
        
        # FFT参数配置
        self.rangeFFTSize = self.numADCSamples  # 范围FFT大小
        self.dopplerFFTSize = self.numChirpsPerTX  # 多普勒FFT大小
        self.angleFFTSize = 180  # 角度FFT大小
        self.angleGridDeg = np.linspace(-90, 90, 1801)
        
        # 性能参数
        self.maxRange = 8.792
        self.maxDoppler = 4.435
        self.rangeRes = self.c / (2 * self.bandWidthValid)
        self.dopplerRes = 0.069
        
        self.scale = np.arange(-60, 60.2, 0.2)
        self.slowFs = 1 / self.Tr
        self.indexSpace = np.linspace(-self.numChirpsPerTX / 2, self.numChirpsPerTX / 2, self.numChirpsPerTX)
        self.dopplerIndex = self.indexSpace * (self.slowFs * self.lambda_) / (self.numChirpsPerTX*2)
    
    def printParams(self):
        # 打印所有雷达参数
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
        # 打印所有雷达参数
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}")

class RD_Grid_Params:
    def __init__(self,radarParams):
        self.rangeInterval = radarParams.rangeRes
        self.dopplerInterval = radarParams.dopplerRes
        self.color = 'r' # 红色网格线
        self.lineStyle = '-.' # 点划线样式
        self.lineWidth = 1 # 线宽为1
    def printParams(self):
        # 打印所有雷达参数
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}")

class PointCloud_detection_visual:
    def __init__(self):
        super().__init__()
    
    def range_heatmap(self, rdMatrix, param, frameIdx, rxIdx, logScale=True):
        # 选择特定天线和帧的数据
        data = rdMatrix[rxIdx, :, :, frameIdx]
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
    def doppler_heatmap(self, rdMatrix, param, frameIdx, rxIdx, logScale=True, gridParams=None, title1=None, title2=None):
        # 提取指定天线和帧的数据
        data = np.squeeze(rdMatrix[rxIdx, :, :, frameIdx])
        # 计算功率谱
        powerSpectrum = np.abs(data)**2
        # 是否以dB绘图
        if logScale:
            powerSpectrum = 10 * np.log10(powerSpectrum + 1e-10)
        # 构建坐标轴
        rangeBins = np.arange(param.rangeFFTSize)
        rangeAxis = rangeBins * param.rangeRes
        dopplerBins = np.arange(param.dopplerFFTSize)
        dopplerAxis = (dopplerBins - param.dopplerFFTSize/2) * param.dopplerRes
        # 绘制热力图
        plt.figure()
        # 使用imshow绘制热力图，注意转置匹配MATLAB的imagesc行为
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
        
            # 如果提供了网格参数，则绘制网格线
        if gridParams is not None:
            # 绘制距离网格线
            max_range = np.max(rangeAxis)
            range_ticks = np.arange(0, max_range + gridParams.rangeInterval, gridParams.rangeInterval)
            for r in range_ticks:
                plt.plot([r, r], [np.min(dopplerAxis), np.max(dopplerAxis)],
                        color=gridParams.color,
                        linestyle=gridParams.lineStyle,
                        linewidth=gridParams.lineWidth)
            
            # 绘制速度网格线
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
        self.d = radarParams.d # 天线阵列间距
        self.numADCSamples = radarParams.numADCSamples
        self.numChirpsPerTX = radarParams.numChirpsPerTX
        self.angleFFTSize = radarParams.angleFFTSize
        self.rangeFFTSize = radarParams.rangeFFTSize
        self.dopplerFFTSize = radarParams.dopplerFFTSize
    def printParams(self):
        # 打印所有雷达参数
        for key, value in self.__dict__.items():
            keyName = key.replace("_", " ").capitalize()
            print(f"{keyName}: {value}") 

def topN_rd(dopplerCoherentPerFrame, nPeaks=10):
    """保留RD图中N个最强幅度的像素点, 返回掩码和对应的复数值

    Args:
        dopplerCoherentPerFrame (_type_): 输入的复数矩阵，代表每帧的多普勒相干数据
        nPeaks (int, optional): 要保留的最强像素点的数量. Defaults to 10.
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
    """根据多普勒相干数据和RD图峰值掩码生成点云(距离、速度、方位角、俯仰角)

    Args:
        dopplerOutResultPerFrame (_type_): dopplerFFT后的复数数据
        rdTopNMask (_type_): 2D bool 数组(rangeFFTSize, dopplerFFTSize), RD图中前N个峰值的掩码
        doaParams (_type_): 参数字典
    Returns:
        results (_type_): 2D数组(numTargets x 4), 每行存储一个目标的物理参数：
                        [range(m), velocity(m/s), azimuth(rad), elevation(rad)]
    """
    # -------------------------- 1. 初始化输出与中间数组 --------------------------
    # 点云结果初始化(空数组,后续动态添加目标)
    Results = np.zeros((0, 4), dtype=np.float64)
    Results_n = 0
    # 俯仰角数组(2通道x距离binx多普勒bin)
    elevationArrays = np.zeros((2, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # 方位角数组(numAnt通道x距离binx多普勒bin)
    azimuthArrays = np.zeros((doa_params.A_n, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # -------------------------- 2. 填充俯仰角/方位角原始数据 --------------------------
    # 为俯仰角数组赋值 关键：MATLAB 1-based索引 → Python 0-based索引（5→4，3→2）
    elevationArrays[0, :, :] = dopplerOutResultPerFrame[4, :, :]
    elevationArrays[1, :, :] = dopplerOutResultPerFrame[2, :, :]
    
    # 为方位角数组赋值（分两段：1-4通道、5-8通道，对应原MATLAB逻辑）
    for vn in range(4):  # 原Vn=1~4 → Python vn=0~3
        azimuthArrays[vn, :, :] = dopplerOutResultPerFrame[vn, :, :]
    for vn in range(4,8): # 原Vn=5~8 → Python vn=4~7，对应原Vn+4（5+4=9→Python 8）
        azimuthArrays[vn, :, :] = dopplerOutResultPerFrame[vn + 4, :, :]
    # -------------------------- 3. 筛选目标点（非目标点数据置零） --------------------------
    # 提取非目标点 / 目标点的行列索引 
    comR, comD = np.where(rdTopNMask == False)  # 非目标点
    # print(f"nonTargetR.shape: {nonTargetR.shape}, nonTargetR: {nonTargetR}")
    r1, c1 = np.where(rdTopNMask == True)  # 目标点
    # print(f"targetR: {r1}, targetD: {c1}")
    # 生成非目标点的扁平化掩码（原MATLAB sub2ind功能）
    mask = np.zeros((doa_params.rangeFFTSize, doa_params.dopplerFFTSize),dtype=bool)
    mask[comR, comD] = True
    
    # 非目标点数据置0(俯仰角)
    for vn in range(2):
        layer = np.squeeze(elevationArrays[vn,:,:].copy())
        layer[mask] = 0 
        elevationArrays[vn,:,:] = layer
    # 非目标点数据置0(方位角)
    for vn in range(doa_params.A_n):
        layer = np.squeeze(azimuthArrays[vn,:,:].copy())
        layer[mask] = 0
        azimuthArrays[vn,:,:] = layer
    # -------------------------- 4. 角度FFT计算（方位角+俯仰角） --------------------------
    # 初始化角度FFT结果存储数组
    # 方位角FFT谱
    sigAfft = np.zeros((doa_params.angleFFTSize, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # 俯仰角FFT谱
    sigEfft = np.zeros((doa_params.angleFFTSize, doa_params.rangeFFTSize, doa_params.dopplerFFTSize), dtype=complex)
    # 存储每个目标点的角度FFT峰值索引
    angleA = np.zeros(len(r1), dtype=np.int64)  # 方位角峰值索引
    angleE = np.zeros(len(r1), dtype=np.int64)  # 俯仰角峰值索引
    index = 0 # 设置索引
    # 逐个目标点计算角度FFT
    for i in range(len(c1)):
        row = r1[i]  # 当前目标的距离bin（0-based）
        col = c1[i]  # 当前目标的多普勒bin（0-based）
                       # 逐个目标点提取索引

        # 方位角FFT（补零+频域移位）
        tempA = azimuthArrays[:, row, col]  # 提取目标点8个接收通道的复数数据
        temp_fftA = fftshift(fft(tempA, doa_params.angleFFTSize))
        sigAfft[:, row, col] = temp_fftA
        maxAIdx = np.argmax(np.abs(temp_fftA))  # 寻找角度FFT后最大幅度值点的索引
        angleA[index] = maxAIdx # 存储最大幅度值点的索引
        
        # 俯仰角FFT（补零+频域移位）
        tempE = elevationArrays[:, row, col]  # 提取该点2个俯仰角通道数据
        tempE_fftE = fftshift(fft(tempE, n=doa_params.angleFFTSize))
        sigEfft[:, row, col] = tempE_fftE
        maxEIdx = np.argmax(np.abs(tempE_fftE))
        angleE[index] = maxEIdx # 存储最大幅度值点的索引
        
        index += 1
    
    # -------------------------- 5. 解算目标物理参数 --------------------------
    # 解算距离、方位角、俯仰角
    R_idx = -1  # 初始化为-1，因为索引从0开始
    Azimuth = 0
    Elevation = 0
    Elevation_Phi = np.zeros((2, 0))  # 初始化为空数组
    Elevation_Phi_diff = 0

     # 初始化Results数组，预分配空间
    num_targets = len(r1)
    Results = np.zeros((num_targets, 4))
    Results_n = 0
    
    for i in range(len(r1)):
        # 如果这一行已经在上一次处理过就跳过，避免重复
        if r1[i] == R_idx:
            continue

        A_idx = angleA[i]  # 该目标在方位角谱上的峰值Bin
        E_idx = angleE[i]  # 该目标在俯仰角谱上的峰值Bin
        R_idx = r1[i]  # Range_bin 像素所在距离轴索引
        C_idx = c1[i]  # Doppler_bin 像素所在多普勒轴索引

        # 扩展Elevation_Phi数组以容纳新目标
        if Results_n >= Elevation_Phi.shape[1]:
            # 动态扩展数组
            new_cols = np.zeros((2, Elevation_Phi.shape[1] + 10))
            if Elevation_Phi.shape[1] > 0:
                new_cols[:, :Elevation_Phi.shape[1]] = Elevation_Phi
            Elevation_Phi = new_cols

        # 计算相位角
        for Rn in range(2):
            Elevation_Phi[Rn, Results_n] = np.angle(elevationArrays[Rn, R_idx, C_idx])

        Elevation_Phi_diff = Elevation_Phi[0, Results_n] - Elevation_Phi[1, Results_n]
        Elevation_Phi_diff = np.mod(Elevation_Phi_diff + np.pi, 2 * np.pi) - np.pi
        
        # 按照公式计算相关参数
        Fb = ((R_idx) * doa_params.Fs) / doa_params.numADCSamples  # 注意：Python索引从0开始
        Fd = (C_idx - doa_params.numChirpsPerTX / 2) / (doa_params.numChirpsPerTX * doa_params.Tr)  # 多普勒频率
        Fa = (A_idx - doa_params.angleFFTSize / 2) / doa_params.angleFFTSize  # 方位角频率
        Fe = (E_idx - doa_params.angleFFTSize / 2) / doa_params.angleFFTSize  # 俯仰角频率
    
        R = doa_params.c * Fb / (2 * doa_params.slope)  # 距离解算
        V = doa_params.lambda_ * Fd / 2  # 速度解算，注意：lambda是Python关键字
        Azimuth = np.arcsin(Fa * doa_params.lambda_ / doa_params.d)  # 方位角度解算
        Elevation = np.arcsin(Elevation_Phi_diff * doa_params.lambda_ / (2 * np.pi * doa_params.d))
    
        # 存储结果
        Results[Results_n, 0] = R
        Results[Results_n, 1] = V
        Results[Results_n, 2] = Azimuth
        Results[Results_n, 3] = Elevation
    
        Results_n += 1

    return Results

filePath = "/Volumes/系统/dataset/2025_07_22/adc_data.bin"

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
    
    # 初始化全0数组
    rangeFFT_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames),dtype=complex)
    DopplerFFT_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames), dtype=complex)
    MTI_out_result = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize, radarParams.numFrames), dtype=complex)
    # 逐帧分析
    for nF in range(radarParams.numFrames):
        print(f"Processing frame {nF+1} of {radarParams.numFrames}")
        # 生成汉宁窗
        rangeWindowCoeffVec = np.hanning(radarParams.rangeFFTSize)
        # 初始化存储数组
        rangeFFT_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.numChirpsPerTX), dtype=complex)
        # range FFT
        for i_an in range(radarParams.numAnt):
            # 提取对应维度的数据。squeeze移除单维度
            RangeMatMeanBefore = np.squeeze(virtualAntennaArrayAfter[i_an,:,:,nF])
            # 去除均值（静态杂波去除）
            # MATLAB中mean默认按列计算（维度2），python中需指定axis=0
            RangeMatMeanAfter = RangeMatMeanBefore - np.mean(RangeMatMeanBefore, axis=0, keepdims=True)
            # 应用窗函数
            # repmat 在python中可用np.tile替代，注意维度匹配
            # rangeWindowCoeffVec[:, np.newaxis]:将一维的窗函数系数向量从（N,）转换为（N,1）np.tile():用于重复数组；(1, radarParams.numChirpsPerTX)：1表示在第一个维度上不重复，第二个参数表示在第二个维度重复指定次数
            # 结果：将窗函数从系数（N,1）扩展为（N,numChirpsPerTX）二维数组
            RangeMat_window_after = RangeMatMeanAfter * np.tile(rangeWindowCoeffVec[:, np.newaxis], (1, radarParams.numChirpsPerTX))
            # 执行FFT，指定在第一个维度上进行 (axis=0对应MATLAB的1)
            fftOutput_Range = np.fft.fft(RangeMat_window_after, radarParams.rangeFFTSize, axis=0)
            
            # 存储结果
            rangeFFT_out_idx[i_an, :, :] = fftOutput_Range
            # print(f"fftOutput_Range.shape: {fftOutput_Range.shape}")
        # print(f"rangeFFT_out_idx.shape: {rangeFFT_out_idx.shape}")
        
        
        DopplerWindowCoeffVec = np.hanning(radarParams.dopplerFFTSize)
        # 预分配输出数组（复数类型）
        DopplerFFT_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize),dtype=complex)
        # doppler FFT
        for i_an in range(radarParams.numAnt):
            # 提取二维切片 [rangeFFTSize x numChirpsPerTX]
            DopplerMatWinBefore = np.squeeze(rangeFFT_out_idx[i_an, :, :])
            DopplerMatWinAfter = DopplerMatWinBefore * np.tile(DopplerWindowCoeffVec[:, np.newaxis].T,(radarParams.rangeFFTSize,1))
            # 执行多普勒FFT
            fftOutputDoppler = np.fft.fft(DopplerMatWinAfter, radarParams.dopplerFFTSize, axis=1)
            # print(fftOutput_Doppler.shape)
            # 傅里叶移位（第二个维度）
            fftOutputDopplerShift = np.fft.fftshift(fftOutputDoppler, axes=1)
            # print("fftOutputDopplerShift.shape:",fftOutputDopplerShift.shape)
            # 存储结果
            DopplerFFT_out_idx[i_an, :, :] = fftOutputDopplerShift
        
        # MTI
        if nF > 0:
            MTI_out_idx = np.zeros((radarParams.numAnt, radarParams.rangeFFTSize, radarParams.dopplerFFTSize),dtype=complex)
            # 对每个距离单元、多普勒单元和天线应用MTI滤波器
            for i_an in range(radarParams.numAnt):
                for i_range in range(radarParams.rangeFFTSize):
                    for i_doppler in range(radarParams.dopplerFFTSize):
                        # 获取当前帧的RD谱数据
                        currentRdData = DopplerFFT_out_idx[i_an,i_range,i_doppler]
                        # 获取上一帧相同位置的RD谱数据
                        previousRdData = DopplerFFT_out_result[i_an,i_range,i_doppler,nF-1]
                        # 应用MTI滤波器(一次对消：当前帧-上一帧)
                        MTI_out_idx[i_an,i_range,i_doppler] = currentRdData - previousRdData
            
            # 保存MTI处理后的数据
            MTI_out_result[:, :, :, nF] = MTI_out_idx
        else:
            # 第一帧直接保存原始DopplerFFT结果
            MTI_out_result[:, :, :, nF] = DopplerFFT_out_idx

        # 保存中间结果
        rangeFFT_out_result[:, :, :, nF] = rangeFFT_out_idx
        DopplerFFT_out_result[:, :, :, nF] = DopplerFFT_out_idx

    # 循环结束
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
        # 原始Doppler热图(第一个天线通道)
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
    
    # 对天线维度进行相干合并，提升SNR
    DopplerCoherent = np.zeros((radarParams.rangeFFTSize, radarParams.numChirpsPerTX,radarParams.numFrames),dtype=complex)
    for i_nF in range(radarParams.numFrames):
        for i_channel in range(radarParams.numAnt):
            # 累加MTI处理后的结果
            DopplerCoherent[:, :, i_nF] += MTI_out_result[i_channel, :, :, i_nF]
    
    # 处理帧10到20(MATLAB的10:20对应python的9:20)
    for i_nF in range(9,20):
        # 获取top10掩码和复数数据
        rdTopNMask, rdTopNComplex = topN_rd(DopplerCoherent[:, :, i_nF],nPeaks=10)
        Results = generatePointCloud(dopplerOutResultPerFrame=MTI_out_result[:, :, :, i_nF], 
                                rdTopNMask=rdTopNMask, 
                                doa_params=doa_params)
        print(Results.shape)
        # 截取实际结果
        R = Results[:, 0]
        v = Results[:, 1]
        azimuth = np.rad2deg(Results[:,2])
        elevation = np.rad2deg(Results[:,3])
        # 可视化部分
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(azimuth, R, elevation, c=v, cmap='jet', s=20)
        plt.colorbar(scatter)
        
        ax.set_ylabel('Range (m)')
        ax.set_xlabel('Azimuth (°)')
        ax.set_zlabel('Elevation (°)')
        ax.set_title(f'Point-cloud: Range–Az–El (colour = speed) - Frame {i_nF}')
        
        # 设置坐标轴范围
        ax.set_ylim([0, (radarParams.rangeFFTSize - 1) * radarParams.rangeRes])
        ax.set_xlim([-90, 90])
        ax.set_zlim([-60, 60])
        
        plt.grid(True)
        plt.show()