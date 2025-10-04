import sys
import numpy as np
import pdb
import math
import copy
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# from IPython.core.interactiveshell import InteractiveShell

# 设置Numpy数组完整显示(不省略中间部分)
np.set_printoptions(threshold=np.inf)

class PointCloud_detection_utils:
    def __init__(self):
        super().__init__()
    """
    读取二进制文件
    :param filename: 二进制文件路径
    :param param: 毫米波雷达参数
    :return Virtual_Antenna_Arrays 读取的二进制数据
    """
    def bin_to_IQ(self, filePath, param):

        adcDataRow = np.fromfile(filePath, dtype = np.int16)
        fileSize = len(adcDataRow)
        print(f"file size: {fileSize}")
        # 位宽校正
        if param.numADCBits != 16:
            l_max = 2 ** (param.numADCBits - 1) - 1
            adcDataRow[adcDataRow > l_max]  = adcDataRow[adcDataRow > l_max] - 2 ** param['numADCBits']
        adcData = adcDataRow[:fileSize]
        print(f"adcData size: {len(adcData)}")

        # 实复数采样
        if param.isReal:
            # 实数采样处理
            numChirps = fileSize // param.numAdcSamples // param.numRX
            LVDS = adcData.reshape((param.numAdcSamples * param.numRX, numChirps), order='F').T
        else:
            # 复数采样处理
            numChirps = fileSize // 2 // param.numADCSamples // param.numRX // param.numFrames
            print(f"numChirps: {numChirps}")
            LVDS = np.zeros(fileSize // 2, dtype = np.complex64)
            print(f"LVDS size: {len(LVDS)}")
            
            counter = 0
            for i in tqdm(range(0, fileSize - 1, 4),desc="复数采样处理"):
                LVDS[counter] = adcData[i] + 1j * adcData[i + 2]
                LVDS[counter + 1] = adcData[i + 1] + 1j * adcData[i + 3]
                counter += 2
            LVDS = LVDS.reshape((param.numADCSamples * param.numRX, numChirps * param.numFrames),order='F').T
            print(f"LVDS shape: {LVDS.shape}")
        
        # 数据按接收天线分组
        adcData = np.zeros((param.numRX, numChirps * param.numFrames * param.numADCSamples), dtype=np.complex64)
        # print(f"adcData shape: {adcData.shape}")
        for row in tqdm(range(param.numRX), desc="数据按接收天线分组"):
            for i in range(numChirps * param.numFrames):
                adcData[row, i * param.numADCSamples : (i+1) * param.numADCSamples] = LVDS[i, row * param.numADCSamples : (row+1) * param.numADCSamples]    
                
        # 重组为3D矩阵
        sigReceive = np.zeros((param.numRX, param.numADCSamples, numChirps * param.numFrames), dtype=LVDS.dtype)
        for row_r in tqdm(range(param.numRX),desc="重组为3D矩阵"):
            sigReceive[row_r, :, :] = adcData[row_r, :].reshape((param.numADCSamples, numChirps * param.numFrames),order='F')       
            
        # 虚拟天线阵列重组（MIMO处理）
        Virtual_Antenna_Arrays_before = np.zeros((param.numTX * param.numRX, param.numADCSamples, numChirps * param.numFrames // param.numTX), dtype=np.complex64)
        for Rn in tqdm(range(param.numRX), desc= "虚拟天线阵列重组"):
            Cv = 0
            for Cn in range(0, numChirps * param.numFrames, param.numTX):
                for nT in range(param.numTX):
                    Tn = nT * param.numRX
                    Virtual_Antenna_Arrays_before[Tn + Rn, :, Cv] = sigReceive[Rn, :, Cn + nT]
                Cv += 1
        # print(Virtual_Antenna_Arrays_before[1,:,1])
        Virtual_Antenna_Arrays_after = np.zeros((param.numRX * param.numTX, param.numADCSamples, numChirps // param.numTX, param.numFrames), dtype=np.complex64)
        for nF in range(param.numFrames):
            for nC in range(numChirps // param.numTX):
                Virtual_Antenna_Arrays_after[:,:,nC,nF] = Virtual_Antenna_Arrays_before[:,:,nC + nF * numChirps // param.numTX]
        # print(Virtual_Antenna_Arrays_after[0,:,1,0])
        # before = Virtual_Antenna_Arrays_before[:,:,20]
        # after = Virtual_Antenna_Arrays_after[:,:,0,1]
        # print(np.allclose(before, after))
        return Virtual_Antenna_Arrays_after
    """
    逐帧RangeFFT
    :param input: 输入某帧数据
    :param param: 毫米波雷达参数
    :return: RangeFFT后的结果
    """
    def rangeFFT(self, input, param):
        """
        Performs range FFT on the input data using the specified options.

        :param input: Input data array (numpy array)
        :param opt: Options including range_fftsize
        :return: Array with applied range FFT (numpy array)
        """
        range_fftSize = param['range_fftSize']
        # 生成一个长度比原始数据大2的窗函数，再对扩展后的窗函数进行切片，去除首尾的0值点
        # 经过这样的处理，得到的窗函数在边缘处的值不再是0，而是一个较小的正数，从而保留了边缘的部分能量
        rangeWindowCoeffVec = np.hanning(input.shape[1] + 2)[1:-1]
        numChirps = input.shape[2]
        numAnt = input.shape[0]
        out = np.zeros((numAnt, range_fftSize, numChirps), dtype=complex)

        for i_an in range(numAnt):
            inputMat = input[i_an,:,:]
            inputMat = np.subtract(inputMat, np.mean(inputMat, axis=0))
            inputMat = inputMat * rangeWindowCoeffVec[:, None]
            fftOutput = np.fft.fft(inputMat, range_fftSize, axis=0)
            # pdb.set_trace()
            out[i_an, :, :] = fftOutput    
        return out
    
    """
       均值向量相消法：去除静态杂波
    """
    def Vector_mean_cancel(self, rangeFFT_out_idx):
        """
        参数：
        ----------
        rangeFFT_out_idx :
            距离FFT后的复数数据, 形状为 (numAntennas,range_fftSize,numChirpsPerTX)。
        返回：
        ----------
        rangeFFT_out_vmsData :
            应用VMS后的数据, 形状与 rangeFFT_out_result 相同。
            
        处理流程：
            1. 沿chirp维度计算均值(静态背景估计)
            2. 原始信号减去静态背景估计值
            3. 保留动态目标信号，抑制静止杂波
        """
        # 沿脉冲(Chirp)维度(numChirpsPerFrame)计算向量均值。保留维度用于广播计算 (keepdims=True: 保持结果的维度与输入数组一致)
        vectorMean = np.mean(rangeFFT_out_idx, axis=2, keepdims=True)
        # print(f"mean_vector.shape: {vectorMean.shape}")
        # 均值相消
        rangeFFT_out_vmsData_idx = rangeFFT_out_idx - vectorMean
        # print(f"vms_data.shape: {vmsData.shape}")

        return rangeFFT_out_vmsData_idx
    """
    逐帧DopplerFFT
    :param input: 输入某帧数据
    :param param: 毫米波雷达参数
    :return: DopplerFFT后的结果
    """
    def dopplerFFT(self, input, param):
        """
        Performs Doppler FFT on the input data using the specified options.

        :param input: Input data array (numpy array)
        :param opt: Options including doppler_fftsize
        :return: Array with applied Doppler FFT (numpy array)
        """
        fftSize = param['doppler_fftSize']
        dopplerWindowCoeffVec = np.hanning(input.shape[2] + 2)[1:-1]
        numADCSamples = input.shape[1]
        numAnt = input.shape[0]
        out = np.zeros((numAnt, numADCSamples, fftSize), dtype=complex)

        for i_an in range(numAnt):
            inputMat = np.squeeze(input[i_an, :, :])
            inputMat = inputMat * dopplerWindowCoeffVec[None, :]
            fftOutput = np.fft.fft(inputMat, fftSize, axis=1)
            fftOutput = np.fft.fftshift(fftOutput, axes=1)
            out[i_an, :, :] = fftOutput

        return out
    """
        MVDR 波束成形
        :param RangeFFT_out_result: RangeFFT结果
        :param param: 毫米波雷达参数
        :param angle_grid_deg: 角度扫描网格(度数)
        :param frame_idx: 帧索引
    """
    def MVDR_beamforming(self, RangeFFT_out_result, param,frame_idx, log_scale = True):
        # 累加多个chirp提升信噪比
        signal = np.mean(RangeFFT_out_result[:8,:,:,frame_idx], axis=2) # 维度(numAntennas, Range_fftSize)
        numAntennas, range_bins = signal.shape
        print(f"MVDR numAntennas: {numAntennas} range_bins: {range_bins}")
        angle_samples = len(param['angle_grid_deg']) # 角度采样点数
        angle_grid = np.deg2rad(param['angle_grid_deg']) # 转换为弧度
        angle_spectrum = np.zeros((angle_samples, range_bins),dtype=np.float32) # 角度谱
        for rb in range(range_bins):
            # 提取当前距离单元的信号(各天线的复振幅)
            x = signal[:, rb]
            
            # 计算协方差矩阵(加正则化避免奇异性)
            R = np.cov(x, rowvar=True) + 1e-6 * np.eye(numAntennas) #(numAntennas,numAntennas) 8
            
            # 构建导向矢量(均匀线阵导向矢量)
            a = np.exp(1j * 2 * np.pi * param['d'] / param['lambda'] * np.arange(numAntennas)[:,np.newaxis] * np.sin(angle_grid)) #(numAntennas,angle_samples) 8*180
            
            # 计算MVDR权值
            R_inv = np.linalg.inv(R) #(numAntennas,numAntennas) 8*8
            numerator = R_inv @ a
            denominator = np.sum(a.conj() * (R_inv @ a), axis=0)
            w = numerator / denominator #(numAntennas,angle_samples) 8*180
            # 计算角度谱（能量归一化
            spectrum = np.abs(w.T @ x) ** 2 #(angle_samples,) 180
            spectrum_normalized = spectrum / np.max(spectrum) if np.max(spectrum) != 0 else spectrum
            if log_scale:
                angle_spectrum[:,rb] = 20 * np.log10(spectrum_normalized + 1e-10) #转换为dB
            else:
                angle_spectrum[:,rb] = spectrum_normalized
        return angle_spectrum
        
    """
    CFAR(恒虚警率)算法的雷达目标检测函数，实现二维(距离+多普勒)的联合检测
    :param input: 输入某帧数据
    :param param: 毫米波雷达参数
    :param debug: 是否开启调试模式, 默认为False
    :return: 检测结果
    """
    def CFAR(self, input, CFAR_param, debug=False):
        # 非相干积累：将同一距离门(采样点)、同一脉冲序列下不同天线接收到的信号功率累加，得到维度为(采样点数、脉冲数)的数组，每个值代表对应距离-时间点的总接收功率
        # 提高信噪比
        print(f"input_before: {input.shape}")
        sig_integrate = np.sum(np.abs(input) ** 2, axis=0) + 1
        print(f"input_after: {sig_integrate.shape}")
        
        if debug:
            pdb.set_trace()
        # 返回检测到的目标数量、目标的距离索引、噪声估计值和SNR
        N_obj_Rag, Ind_obj_Rag, noise_obj, CFAR_SNR = self.CFAR_CASO_Range(sig_integrate, CFAR_param)
        print(f"在距离维进行检测后的目标数量 N_obj_Rag: {N_obj_Rag}")
        # print(f"在距离维进行检测后的目标 Ind_obj_Rag: {Ind_obj_Rag}")
        detection_results = {}
        # 如果检测到的目标数量大于0，进行第二级CFAR检测(多普勒维度)
        if N_obj_Rag > 0:
            # 在多普勒维度上进行重叠检测，进一步确定目标
            N_Obj, Ind_obj, noise_obj_an = self.CFAR_CASO_Doppler_overlap(sig_integrate,Ind_obj_Rag,input,CFAR_param)
            print(f"在多普勒维进行重叠检测后的目标数量 N_obj: {N_Obj}")
            # print(f"在多普勒维进行重叠检测后的目标索引 Ind_obj: {Ind_obj}")
            detection_results = {}
            #存放每个目标对应的噪声方差
            noise_obj_agg = []
            # 处理每个检测到的目标
            for i_obj in range(1, N_Obj+1):
                try:
                    index1R = Ind_obj[i_obj-1,0] # 获取第i个目标的“距离”索引
                    index1D = Ind_obj[i_obj-1,1] # 获取第i个目标的“多普勒”索引
                except:
                    # 如果只有一个目标 将其reshape成二维数组
                    Ind_obj = Ind_obj.reshape([1,-1]) # 将[R,D] reshape 成[[R,D]]
                    index1R = Ind_obj[i_obj-1,0] # 获取第i个目标的“距离”索引
                    index1D = Ind_obj[i_obj-1,1] # 获取第i个目标的“多普勒”索引
                        
                ind2R = np.argwhere(Ind_obj[:,0] == index1R)
                ind2D = np.argwhere(Ind_obj[ind2R,1] == index1D)
                noiseInd = ind2R[ind2D[0][0],ind2D[0][1]].squeeze()
                if noiseInd.size != 0:
                    noise_obj_agg.append(noise_obj[noiseInd])
                else:
                    print("noiseInd is empty")
            
            print(f"len of noise_obj_agg: {len(noise_obj_agg)} noise_obj_agg: {noise_obj_agg}")
                    
                    
            for i_obj in range(1,N_Obj + 1):
                xind = (Ind_obj[i_obj-1,0] -1) + 1
                detection_results[i_obj] = {'rangeInd':Ind_obj[i_obj-1, 0] - 1} # 距离索引
                # 需要+1
                detection_results[i_obj]['range'] = (detection_results[i_obj]['rangeInd'] + 1) * CFAR_param['rangeBinSize']  # 实际距离
                dopplerInd  = Ind_obj[i_obj-1, 1] - 1 # Doppler索引
                detection_results[i_obj]['dopplerInd_org'] = dopplerInd # Doppler索引
                detection_results[i_obj]['dopplerInd'] = dopplerInd # Doppler索引
        
                # velocity estimation
                detection_results[i_obj]['doppler'] = (dopplerInd + 1 - CFAR_param['doppler_fftSize']/2) * CFAR_param['velocityBinSize'] # 实际速度 包含正负方向
                detection_results[i_obj]['doppler_corr'] = detection_results[i_obj]['doppler'] # 实际速度 包含正负方向(理论上来说是修正后的)

                detection_results[i_obj]['noise_var'] = noise_obj_agg[i_obj-1] # 噪声值

        
                detection_results[i_obj]['bin_val']  = np.reshape(input[:, xind, Ind_obj[i_obj-1,1]],(CFAR_param['numAntenna'],1),order='F')  # 2d FFT value for the 4 antennas

                # 
                # detection_results(i_obj).estSNR  = 10*log10(sum(abs(detection_results (i_obj).bin_val).^2)/sum(detection_results (i_obj).noise_var));  %2d FFT value for the 4 antennas
                detection_results[i_obj]['estSNR']  = (np.sum(np.abs(detection_results[i_obj]['bin_val'] ** 2))/np.sum(detection_results[i_obj]['noise_var'])) 
        
                sig_bin = np.array([[]]).transpose()
                # only apply max velocity extention if it is enabled and distance is larger than minDisApplyVmaxExtend
                if ((CFAR_param['applyVmaxExtend'] == 1) and (detection_results[i_obj]['range'] > CFAR_param['minDisApplyVmaxExtend']) and (len(CFAR_param['overlapAntenna_ID']))):
                    raise Exception("NotCompleted Error")
                else:
                    # Doppler phase correction due to TDM MIMO without apply Vmax extention algorithm
                    deltaPhi = 2*(math.pi)*(dopplerInd + 1 - CFAR_param['doppler_fftSize'] / 2)/( CFAR_param['numTxChan'] * CFAR_param['doppler_fftSize'])
                    sig_bin_org = detection_results[i_obj]['bin_val']
                    for i_TX in range(1,CFAR_param['numTxChan'] + 1):
                        RX_ID = np.linspace((i_TX-1)*CFAR_param['numRxChan']+1, i_TX*CFAR_param['numRxChan'], i_TX*CFAR_param['numRxChan']-(i_TX-1)*CFAR_param['numRxChan'])

                        sig_bin = np.concatenate((sig_bin,sig_bin_org[RX_ID.astype(int) -1]* np.exp(complex(0,(1-i_TX)*deltaPhi))))

                detection_results[i_obj]['bin_val'] = sig_bin
                detection_results[i_obj]['doppler_corr_overlap'] = detection_results[i_obj]['doppler_corr']
                detection_results[i_obj]['doppler_corr_FFT'] = detection_results[i_obj]['doppler_corr']

        return detection_results

                
    def CFAR_CASO_Range(self, sig, CFAR_param):

        '''
         :brief 在距离维上执行CFAR_CASO检测
         :param sig: 2D实值矩阵, 维度为[距离门数, 多普勒门数]
         :param param: 包含 CFAR 参数的字典
         :return N_obj: 检测到的目标数量
         :return Ind_obj: 检测到的目标在 (距离, 多普勒) 维度上的索引数组
        '''
        
        cellNum = CFAR_param['refWinSize'][0]
        gapNum = CFAR_param['guardWinSize'][0]
        K0 = CFAR_param['k0'][0]
        maxEnable = CFAR_param['maxEnable']


        M_samp = sig.shape[0] # 距离门数
        N_pul = sig.shape[1] # 多普勒门数

        # for each point under test, gapNum samples on the two sides are excluded from averaging. Left cellNum/2 and right cellNum/2 samples are used for averaging

        gaptot = gapNum + cellNum
        N_obj = 0
        Ind_obj = []
        noise_obj = []
        CFAR_SNR = []
        
        discardCellLeft = gaptot
        discardCellRight = gaptot

        # for the first gaptot samples only use the right sample

        for k in range(N_pul):
            sigv=np.transpose(sig[:,k])
            vec = sigv[discardCellLeft:(M_samp-discardCellRight)]
            vecLeft = vec[0:(gaptot)]
            vecRight = vec[-(gaptot):]
            # 
            vec = np.concatenate((vecLeft, vec, vecRight))
            for j in range(1,1+M_samp-discardCellLeft-discardCellRight):
            
                cellInd= np.concatenate((np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1) ,np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)))
                cellInd = cellInd + gaptot
                cellInda = np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1)
                cellInda = cellInda + gaptot
                cellIndb = np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)
                cellIndb = cellIndb + gaptot
            
                cellave1a = 0
                for index_cellInda in cellInda:
                
                    cellave1a = cellave1a + vec[int(index_cellInda)-1]
                cellave1a = cellave1a/cellNum

                cellave1b = 0
                for index_cellIndb in cellIndb:
                    cellave1b = cellave1b + vec[int(index_cellIndb)-1]
                cellave1b = cellave1b/cellNum

                cellave1 = min(cellave1a,cellave1b)

                if maxEnable == 1:
                    pass
                else:
                    if vec[j+gaptot-1] > K0*cellave1:
                        N_obj = N_obj+1
                
                        Ind_obj.append([j+discardCellLeft-1, k])
                        noise_obj.append(cellave1) #save the noise level
                        CFAR_SNR.append(vec[j+gaptot-1]/cellave1)

        # get the noise variance for each antenna
        Ind_obj = np.array(Ind_obj)

        for i_obj in range(N_obj):
            # pdb.set_trace()
            ind_range = Ind_obj[i_obj,0]
            ind_Dop = Ind_obj[i_obj,1]
            if ind_range<= gaptot:
                # on the left boundary, use the right side samples twice
                cellInd = np.concatenate((np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum) , np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
            elif ind_range>=M_samp-gaptot+1:
                # on the right boundary, use the left side samples twice
                cellInd = np.concatenate((np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum) , np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum)))
            else:
                cellInd = np.concatenate((np.linspace(ind_range-gaptot, ind_range-gapNum-1,gaptot-gapNum) ,  np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
        
        return N_obj, Ind_obj, noise_obj, CFAR_SNR

    def CFAR_CASO_Doppler_overlap(self, sig_integ,Ind_obj_Rag,sigCpml,CFAR_param):

        '''
        % This function performs 1D CFAR_CASO detection along the Doppler direction, and declare detection only if the index overlap with range detection results. 

        %input
        %   obj: object instance of CFAR_CASO
        %   Ind_obj_Rag: index of range bins that has been determined by the first
        %   step detection along the range direction
        %   sigCpml: a 3D complex matrix, range x Doppler x antenna array
        %   sig_integ: a 2D real valued matrix, range x Doppler 

        %output
        %   N_obj: number of objects detected
        %   Ind_obj: 2D bin index of the detected object
        %   noise_obj_an: antenna specific noise estimation before integration
        '''
        maxEnable = CFAR_param['maxEnable']
        cellNum = CFAR_param['refWinSize'][1]
        gapNum = CFAR_param['guardWinSize'][1]
        K0 = CFAR_param['k0'][1]

        rangeNumBins = sig_integ.shape[0]

        # extract the detected points after range detection
        detected_Rag_Cell = np.unique(Ind_obj_Rag[:,0])
        
        sig = sig_integ[detected_Rag_Cell-1,:]

        M_samp = sig.shape[0]
        N_pul= sig.shape[1]


        # for each point under test, gapNum samples on the two sides are excluded
        # from averaging. Left cellNum/2 and right cellNum/2 samples are used for averaging
        gaptot = gapNum + cellNum

        N_obj = 0
        Ind_obj = np.array([])
        noise_obj_an = []
        vec = np.zeros([N_pul+gaptot*2]) 

        for k in range(1,M_samp+1):
            # get the range index at current range index
            detected_Rag_Cell_i = detected_Rag_Cell[k-1]
            ind1 = np.argwhere(Ind_obj_Rag[:,0] == detected_Rag_Cell_i)
            indR = Ind_obj_Rag[ind1, 1]
            # extend the left the vector by copying the left most the right most
            # gaptot samples are not detected.
            sigv = (sig[k-1,:])
            # pdb.set_trace()
            vec[0:gaptot] = sigv[-gaptot:]
            vec[gaptot: N_pul+gaptot] = sigv
            vec[N_pul+gaptot:] = sigv[:gaptot]
            # start to process
            ind_loc_all = []
            ind_loc_Dop = []
            ind_obj_0 = 0
            noiseEst = np.zeros([N_pul])
            for j in range(1+gaptot,N_pul+gaptot+1):
                cellInd= np.concatenate((np.linspace(j-gaptot, j-gapNum-1,gaptot-gapNum) ,np.linspace(j+gapNum+1,j+gaptot,gaptot-gapNum)))
                noiseEst[j-gaptot-1] = np.sum(vec[ (cellInd-1).astype(int)])

            for j in range(1+gaptot,N_pul+gaptot+1):
                j0 = j - gaptot
                cellInd = np.concatenate((np.linspace(j-gaptot, j-gapNum-1,gaptot-gapNum) ,np.linspace(j+gapNum+1,j+gaptot,gaptot-gapNum)))
                cellInda = np.linspace(j-gaptot,j-gapNum-1,j-gapNum-1-(j-gaptot)+1)
                cellIndb = np.linspace(j+gapNum+1,j+gaptot,j+gaptot-(j+gapNum+1)+1)
                
                cellave1a = 0
                for index_cellInda in cellInda:
                    
                    cellave1a = cellave1a + vec[int(index_cellInda)-1]
                cellave1a = cellave1a/cellNum

                cellave1b = 0
                for index_cellIndb in cellIndb:
                    cellave1b = cellave1b + vec[int(index_cellIndb)-1]
                cellave1b = cellave1b/cellNum

                cellave1 = min(cellave1a,cellave1b)     
                
                maxInCell = np.max(vec[(cellInd-1).astype(int)])

                if maxEnable==1:
                    # detect only if it is the maximum within window
                    condition = ((vec[j-1]>K0*cellave1)) and ((vec[j-1]>maxInCell))
                else:
                    condition = vec[j-1]>K0*cellave1
                
                if condition==1:
                    # check if this detection overlap with the Doppler detection
                    # indR+1 --> differeny index between matlab and python
                    # pdb.set_trace()
                    if np.isin((indR+1).squeeze(), j0).any():
                        # find overlap, declare a detection
                        ind_win = detected_Rag_Cell_i
                        # range index
                        ind_loc_all = ind_loc_all + [ind_win]
                        # Doppler index
                        ind_loc_Dop = ind_loc_Dop + [j0-1]

            ind_loc_all = np.array(ind_loc_all)
            ind_loc_Dop = np.array(ind_loc_Dop)
            

            if len(ind_loc_all)>0:
                ind_obj_0 = np.stack((ind_loc_all,ind_loc_Dop),axis=1)
                if Ind_obj.shape[0] == 0:
                    Ind_obj = ind_obj_0
                else:    
                    # following process is to avoid replicated detection points
                    ind_obj_0_sum = ind_loc_all + 10000*ind_loc_Dop
                    Ind_obj_sum = Ind_obj[:,0] + 10000*Ind_obj[:,1]
                    for ii  in range(1,ind_loc_all.shape[0]+1):
                        if not np.isin(Ind_obj_sum, ind_obj_0_sum[ii-1]).any():
                            Ind_obj = np.concatenate((Ind_obj, ind_obj_0[ii-1,np.newaxis,:]),axis = 0)
        
        N_obj = Ind_obj.shape[0]

        # reset the ref window size to range direction
        cellNum = CFAR_param['refWinSize'][0]
        gapNum = CFAR_param['guardWinSize'][0]
        gaptot = gapNum + cellNum
        # get the noise variance for each antenna
        N_obj_valid = 0
        Ind_obj_valid = []
        
        for i_obj in range(1,N_obj+1):    
            ind_range = Ind_obj[i_obj-1,0]
            ind_Dop = Ind_obj[i_obj-1,1]
            # skip detected points with signal power less than obj.powerThre
            # ToDO: check ind_range和ind_Dop从0开始(同python)?
            
            if (min(np.abs(sigCpml[:, ind_range, ind_Dop]) ** 2) < CFAR_param['powerThre']):
                continue

            if (ind_range+1) <= gaptot:
                # on the left boundary, use the right side samples twice
                cellInd = np.concatenate((np.linspace(ind_range+gapNum+1,ind_range+gaptot, gaptot-gapNum), np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
            elif (ind_range+1) >= rangeNumBins-gaptot+1:
                # on the right boundary, use the left side samples twice
                cellInd = np.concatenate((np.linspace(ind_range-gaptot,ind_range-gapNum-1, gaptot-gapNum), np.linspace(ind_range-gaptot,ind_range-gapNum-1,gaptot-gapNum)))
            else:
                cellInd = np.concatenate((np.linspace(ind_range-gaptot,ind_range-gapNum-1, gaptot-gapNum), np.linspace(ind_range+gapNum+1,ind_range+gaptot,gaptot-gapNum)))
            
            
            N_obj_valid = N_obj_valid +1
            noise_obj_an.append( np.reshape((np.mean(abs(sigCpml[:, cellInd.astype(int), ind_Dop,np.newaxis].copy()) ** 2,axis=1)), (CFAR_param['numAntenna'], 1, 1), order="F"))

            Ind_obj_valid.append( Ind_obj[i_obj-1,:])    
            
        N_obj = N_obj_valid
        Ind_obj = np.array(Ind_obj_valid).squeeze()
        noise_obj_an = np.array(noise_obj_an).squeeze()
        
        return N_obj, Ind_obj, noise_obj_an
    
    def DOA(self, detection_obj, DOA_param, output_est_results = True):
        numObj = len(detection_obj) # 获取检测到的目标数量
        out = copy.deepcopy(detection_obj) # 深拷贝检测结果，用于存储带角度的输出
        numAoAObjCnt = 0 # 记录已估计角度的目标数量
        angle_sepc_2D_fft_dict = {} # 存储二维FFT频谱的字典(键为目标距离)
        
        print(f"检测到的目标数量: {numObj}")
        
        # 遍历每个检测到的目标
        for i_obj in range(1, numObj + 1):
            current_obj = detection_obj[i_obj] # 获取当前目标的信息
            
            if i_obj == 1:
                print(f"当前行号： {sys._getframe().f_lineno + 1 } current_obj: {current_obj}")
            # 计算估计SNR(信号与噪声功率比)
            estSNR = 10 * math.log10(np.sum(np.abs(current_obj['bin_val']) ** 2) / np.sum(current_obj['noise_var']) + 1e-10)
            
            X = current_obj['bin_val'] # 获取目标的频域数据(四根天线的复信号)(二维数组)
            
            # 计算接收信号的协方差矩阵(矩阵与他的共轭转置相乘) 常用于波束形成和DOA估计（未在method=1中使用，可能为其他方法预留）
            R = np.dot(X, X.T.conjugate())

            if DOA_param['method'] == 1: # 方法1：使用二维波束形成FFT方法
                # 若不输出估计结果，则仅返回二维FFT频谱
                if not output_est_results:
                    angle_sepc_2D_fft = self.DOA_beamformingFFT_2D(X, DOA_param, output_est_results=True)
                    current_obj_range = current_obj['range'] # 获取目标的距离
                    print(f"当前目标距离: {current_obj_range}")
                    angle_sepc_2D_fft_dict[current_obj_range] = angle_sepc_2D_fft # 将二维FFT频谱存储到字典中
                    if i_obj == numObj: # 若为最后一个目标，返回字典
                        return angle_sepc_2D_fft_dict
                    else:
                        continue # 跳过后续处理，继续下一个目标
                
                # 获取DOA估计角度和二维FFT频谱
                DOA_angles, angle_sepc_2D_fft = self.DOA_beamformingFFT_2D(X, DOA_param, output_est_results=True)
                
                #初始化输出字典(若为第一个目标
                if numAoAObjCnt == 0:
                    out = {}
                
                # 遍历每个DOA估计结果（可能对应多个角度峰值）
                for i_obj_doa in range(1, DOA_angles.shape[0] + 1):
                    if DOA_angles.size == 0:  # 若未检测到角度，跳出循环
                        break
                
                    numAoAObjCnt += 1  # 已处理目标数+1
                
                    # 构建输出字典，整合原始数据和角度信息
                    out[numAoAObjCnt] = {
                        'rangeInd': current_obj['rangeInd'],
                        'dopplerInd': current_obj['dopplerInd'],
                        'range': current_obj['range'],
                        'doppler_corr': current_obj['doppler_corr'],
                        'dopplerInd_org': current_obj['dopplerInd_org'],
                        'noise_var': current_obj['noise_var'],
                        'bin_val': current_obj['bin_val'],
                        'estSNR': current_obj['estSNR'],
                        'doppler_corr_overlap': current_obj['doppler_corr_overlap'],
                        'doppler_corr_FFT': current_obj['doppler_corr_FFT']
                    }
                    
                    # 提取方位角和仰角（DOA_angles格式为[azim, elev, ...]）
                    out[numAoAObjCnt]["angle_azi"] = DOA_angles[i_obj_doa - 1, 0]
                    out[numAoAObjCnt]["angle_ele"] = DOA_angles[i_obj_doa - 1, 1]
                    
                    # 计算角度对应的SNR并归一化
                    angle_SNR = 10 * np.log10(
                        np.abs(angle_sepc_2D_fft[int(DOA_angles[i_obj_doa - 1, 2]) - 1, int(DOA_angles[i_obj_doa - 1, 3]) - 1]) ** 2 /
                        np.sum(np.abs(angle_sepc_2D_fft)) ** 2 + 1e-10
                    )
                    normalized_snr = (angle_SNR - (-200)) / (0 - (-200))  # 归一化到[0,1]
                    normalized_snr = np.clip(normalized_snr, 0, 1)
                    out[numAoAObjCnt]["SNR_angle"] = normalized_snr
                    
                    # 存储完整角度信息和频谱
                    out[numAoAObjCnt]['angles'] = DOA_angles[i_obj_doa - 1, :]
                    out[numAoAObjCnt]['spectrum'] = angle_sepc_2D_fft
                    
            elif DOA_param['method'] == 2: # 未实现的方法
                raise Exception("NotImplementedError")
            else: # 错误参数处理
                raise Exception("Wrong Parameter method Error")

        return out # 返回整合角度信息后的结果
    
    """
        基于FFT波束成形执行二维角度估计, 方位角峰值选择在一维FFT域中完成，仰角峰值选择在二维FFT之后完成
        :param sig 复信号向量，每个值对应一根天线。该向量的长度等于启用的numTX x numRX;
                   可能存在重叠的天线。此信号需要根据 D 值重新排列，以形成虚拟天线阵列。
        :return angleObj_est 角度估计结果
        :return angle_spec_2D_fft 角度二维FFT频谱
    """
    def DOA_beamformingFFT_2D(self, sig, DOA_param, index = 'a', output_est_results = False):
        # 波束成形视场角
        angles_DOA_az = DOA_param['angles_DOA_az']
        angles_DOA_ele = DOA_param['angles_DOA_el']
        # 天线间距
        d = DOA_param['antDis']
        # 天线坐标2维矩阵
        D = DOA_param['D']
        # 角度FFT点数
        angleFFTSize = DOA_param['angle_fftSize']
        # 首先基于天线坐标系生成一个2D矩阵
        D = np.array(D) + 1
        apertureLen_azim = max(D[:,0]) #方位角孔径长度
        apertureLen_elev = max(D[:,1]) #俯仰角孔径长度
        # print(f"apertureLen_azim: {apertureLen_azim} apertureLen_elev: {apertureLen_elev}")
        sig_2D = np.zeros((apertureLen_azim, apertureLen_elev), dtype = "complex_")
        for i_line in range(1,apertureLen_elev+1):
            ind = np.argwhere(D[:,1] == i_line)
            D_sel = D[ind,0]
            sig_sel = sig[ind]
            value_index, indU = np.unique(D_sel, return_index=True) # -1 to fit matlab's index
    
            sig_2D[D_sel[indU].squeeze() - 1,i_line-1] = sig_sel[indU].squeeze()
            
        # print(f"sig_2D: {sig_2D}")
        # run FFT on azimuth and elevation
        angle_sepc_1D_fft = np.fft.fftshift(np.fft.fft(sig_2D, angleFFTSize, axis=0), axes=0)
        angle_sepc_2D_fft = np.fft.fftshift(np.fft.fft(angle_sepc_1D_fft, angleFFTSize, axis = 1), axes = 1); 
        if not output_est_results:
            return angle_sepc_2D_fft
  
        pi = math.pi
        wx_vec = np.linspace(-pi, pi, angleFFTSize+1)
        wz_vec = np.linspace(-pi, pi, angleFFTSize+1)
        wx_vec = wx_vec[:-1]
        wz_vec = wz_vec[:-1]
        
        spec_azim = np.abs(angle_sepc_1D_fft[:,0])

        if not (index == "a"):
    
            norm = lambda x: (x - x.min())/(x.max() - x.min())
            spec_azim_norm = norm(spec_azim)
            spec_azim_plot = []
            fft_axis = []
            # 
            for index1, value in enumerate(spec_azim_norm):
                theta = math.asin(index1/(angleFFTSize/2) - 1)* 180 / math.pi
                fft_axis.append(theta)
                spec_azim_plot.append(value)
        
        # plt.plot(music_axis,spec_azim_plot, label='FFT')
            plt.plot(fft_axis,spec_azim_plot, label='FFT')
            plt.legend()
            plt.show()
            plt.savefig("AngleFFT_MUSIC_" + str(index) + ".jpg")
            print("AngleFFT_MUSIC_" + str(index) + ".jpg")
            plt.close()
        
        peakVal_azim, peakLoc_azim = self.DOA_BF_PeakDet_loc(spec_azim, DOA_param)

        if apertureLen_elev == 1:
            # 只有方位角没有俯仰角
            obj_cnt = 1
            angleObj_est = []
            for i_obj in range(1, 1+ peakLoc_azim.size):
                ind = peakLoc_azim[i_obj - 1]
      
                azim_est = (math.asin(wx_vec[(ind-1).astype(int)]/(2*pi*d)))/(2*math.pi)*360
                if ((azim_est >= angles_DOA_az[0]) and (azim_est <= angles_DOA_az[1])):
                    angleObj_est.append([azim_est,0,ind,0])
                    obj_cnt = obj_cnt+1
          
                else:
                    continue

            if len(angleObj_est) == 0:
                angleObj_est = np.array([[]])
            else:
                angleObj_est = np.array(angleObj_est)
            
        else:
            # azimuth and elevation angle estimation
            # for each detected azimuth, estimate its elevation 
            obj_cnt = 1
            angleObj_est = []
            for i_obj in range(1, 1+ peakLoc_azim.size):
                ind = peakLoc_azim[i_obj - 1]
                print(f"当前行号: {sys._getframe().f_lineno + 1} ind: {ind}")
                spec_elev = np.abs(angle_sepc_2D_fft[(ind-1).astype(int),:])
                peakVal_elev, peakLoc_elev = self.DOA_BF_PeakDet_loc(spec_elev, DOA_param)
                print(f"当前行号： {sys._getframe().f_lineno + 1 } peakVal_elev: {peakVal_elev} peakLoc_elev: {peakLoc_elev} peakVal_elev.size: {peakVal_elev.size}")
                # calcualte the angle values
                for j_elev in range(1, 1+ peakVal_elev.size):
                    print(f"当前行号：{sys._getframe().f_lineno + 1} {wx_vec[(ind-1).astype(int)]/(2*pi*d)}")
                    print(f"当前行号：{sys._getframe().f_lineno + 1} {wz_vec[peakLoc_elev[j_elev-1].astype(int)-1]/(2*pi*d)}")
                    # 角度
                    azim_est = (math.asin(wx_vec[(ind-1).astype(int)]/(2*pi*d)))/(2*math.pi)*360
                    elev_est = (math.asin(wz_vec[peakLoc_elev[j_elev-1].astype(int)-1]/(2*pi*d)))/(2*math.pi)*360
                    if ((azim_est >= angles_DOA_az[0]) and (azim_est <= angles_DOA_az[1]) and (elev_est >= angles_DOA_ele[0]) and (elev_est <= angles_DOA_ele[1])):
                        angleObj_est.append([azim_est,elev_est,ind[0],peakLoc_elev[j_elev-1][0]])

                        obj_cnt = obj_cnt+1
            
                    else:
                        continue
            if len(angleObj_est) == 0:
                angleObj_est = np.array([[]])
            else:
                angleObj_est = np.array(angleObj_est)
        return angleObj_est, angle_sepc_2D_fft
    def DOA_BF_PeakDet_loc(self, inData, DOA_param):
        gamma = DOA_param['gamma'] # 用于计算峰值阈值的增益因子
        sidelobeLevel_dB = DOA_param['sidelobeLevel_dB'] # 用于计算峰值阈值的侧瓣电平
        inData = inData.squeeze() # 去除输入数据中所有长度为1的维度
        
        minVal = float("inf") # 初始化当前最小值为正无穷
        maxVal = 0 # 初始化当前最大值为0
        maxLoc = 0 # 初始化当前最大值的索引
        maxData = np.array([[]]) # 初始化用于存储检测到的峰信息的空数组
        
        locateMax = 0 # 标记是否处于寻找峰值状态(0 寻找谷值； 1 寻找峰值)
        numMax = 0 # 峰值计数器
        extendLoc = 0                         # 扩展循环索引的长度，用于初始阶段
        initStage = 1                         # 初始阶段标志（1=初始，0=已完成）
        absMaxValue = 0                       # 记录全局最大值
        
        i = 0
        N = inData.size
        while(i < (N + extendLoc -1)):
            i = i + 1
            i_loc = ((i - 1) % N) + 1
            currentVal = inData[i_loc - 1]
            try:
                if currentVal > absMaxValue:
                    absMaxValue = currentVal
            except:
                print(f"currentVal: {currentVal}")
                print(f"absMaxValue: {absMaxValue}")
                raise Exception("Error")
            # record the current max value and location
            if currentVal > maxVal:
                maxVal = currentVal
                maxLoc = i_loc
                maxLoc_r = i
            # record for the current min value and location
            if currentVal < minVal:
                minVal = currentVal
            if locateMax:
                if currentVal < (maxVal / gamma):
                    numMax = numMax + 1
                    bwidth = i - maxLoc_r
                    # Assign maximum value only if the value has fallen below the max by
                    # gamma, thereby declaring that the max value was a peak
                    if maxData.size == 0:
                        maxData = np.concatenate((maxData,np.array([[maxLoc, maxVal, bwidth, maxLoc_r]])), axis=1)
                    else:
                        maxData = np.concatenate((maxData[:numMax-1,:],np.array([[maxLoc, maxVal, bwidth, maxLoc_r]])), axis=0)
                    minVal = currentVal
                    locateMax = 0
            else:
                if currentVal > minVal*gamma:
                # Assign minimum value if the value has risen above the min by
                # gamma, thereby declaring that the min value was a valley
                    locateMax = 1
                    maxVal = currentVal
                    if (initStage == 1):
                        extendLoc = i
                        initStage = 0
        # make sure the max value needs to be cetain dB higher than the side lobes to declare any detection
        estVar = np.zeros((numMax, 1))
        peakVal = np.zeros((numMax, 1))
        peakLoc = np.zeros((numMax, 1))
        delta = []
        maxData_ = np.zeros((numMax, maxData.shape[1]))
        numMax_ = 0
        totPower = 0
        for i in range(1,numMax+1):
            if maxData[i-1, 1] >= (absMaxValue * (pow(10, -sidelobeLevel_dB/10))):
                numMax_ = numMax_ + 1
                maxData_[numMax_ - 1, :] = maxData[i-1, :]
                totPower = totPower + maxData[i-1, 1]

        maxData = maxData_
        numMax = numMax_

        estVar = np.zeros((numMax, 1))
        peakVal = np.zeros((numMax, 1))
        peakLoc = np.zeros((numMax, 1))

        delta = []
        for ind in range(1,numMax+1): 
            peakVal[ind-1] = maxData[ind-1,1]
            peakLoc[ind-1] = ((maxData[ind-1,0]-1) %  N) + 1
        return peakVal, peakLoc    
    
    """
       2D CA-CFAR 检测
    """  
    def CFAR_demo(self, DopplerFFT_out_result, CFAR_param, frame_idx, RX_idx):
        """
            :param DopplerFFT_out_result: 输入的距离-多普勒数组(2维Doppler数组)
            :param CFAR_param: CFAR参数
            :param frame_idx: 输入的帧索引
        """
        Pfa = CFAR_param['Pfa'] # False alarm probability, default 1e-3
        Range_Dim = np.arange(1,CFAR_param['range_fftSize']+1) * CFAR_param['Range_Res'] # 距离
        Doppler_Dim = (np.arange(1,CFAR_param['doppler_fftSize']+1) - CFAR_param['doppler_fftSize'] // 2) * CFAR_param['doppler_res'] # 多普勒
        handleWindow_r = 7
        handleWindow_c = 7
        handleWindow = np.zeros((handleWindow_r, handleWindow_c)) # 处理窗口大小
        proCell_r = 3
        proCell_c = 3
        proCell = np.zeros((proCell_r, proCell_c)) # 处理单元大小
        
        if handleWindow_r % 2 == 0 or handleWindow_c % 2 == 0:
            raise ValueError("Window size must be odd.")
        if proCell_r > handleWindow_r // 2 or proCell_c > handleWindow_c // 2:
            raise ValueError("Guard size must be less than half of the window size.")
        
        # range_doppler_map = DopplerFFT_out_result[RX_idx,:,:,frame_idx] # Merge channels to improve the SNR
        
        # get the size of range-doppler map 
        range_fftSzie, deloppler_fftSize = DopplerFFT_out_result[RX_idx,:,:,frame_idx].shape # 单人 128;56
        
        # Calculate the effective area (to avoid the edges where a complete window cannot be formed)
        CFAR_Map_r = range_fftSzie - (handleWindow_r - 1) # 单人 122
        CFAR_Map_c = deloppler_fftSize - (handleWindow_c - 1) # 单人 50
        
        CFAR_Map = np.zeros((CFAR_Map_r, CFAR_Map_c)) # Initialize CFAR map 122 50
        
        referCellNum = handleWindow_r * handleWindow_c - proCell_r * proCell_c #  the number of reference cells 49-9 = 40
        # 门限因子
        alpha = referCellNum * (Pfa ** (-1 / referCellNum) - 1) # 计算门限
        # 执行参考单元滑动窗口职责
        for i in range(CFAR_Map_r):
            for j in range(CFAR_Map_c):
                # 获取当前处理窗口
                handleWindow = DopplerFFT_out_result[RX_idx,i:i+handleWindow_r, j:j+handleWindow_c,frame_idx] # 获取处理窗口
                # 计算保护单元在窗口中的起始和结束位置
                pro_r_start = int((handleWindow_r - proCell_r) // 2) # 2
                pro_r_end = handleWindow_r - int((handleWindow_r - proCell_r) // 2) # 5
                pro_c_start = int((handleWindow_c - proCell_c) // 2)
                pro_c_end = handleWindow_c - int((handleWindow_c - proCell_c) // 2)
                proCell = handleWindow[pro_r_start:pro_r_end, pro_c_start:pro_c_end] # 获取处理单元
                # 估计噪声均值
                Beta = (np.sum(handleWindow) - np.sum(proCell)) / referCellNum # 计算Beta
                # 判决门限
                CFAR_Map[i,j] = alpha * Beta # 计算CFAR值
        
        CFAR_MapRange_Dim = Range_Dim[:CFAR_Map_r] # 距离维度
        CFAR_MapDoppler_Dim = Doppler_Dim[:CFAR_Map_c] # 多普勒维度
        
        #---------------------------------------可视化-------------------------------------
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111,projection='3d')
        
        # 绘制3D网格图
        X, Y = np.meshgrid(Range_Dim, Doppler_Dim)
        Z = 10 * np.log10(DopplerFFT_out_result[RX_idx,:,:,frame_idx].T)  # 转置以匹配MATLAB的mesh行为
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        
        # 设置坐标轴标签和标题
        ax1.set_xlabel('Range/m')
        ax1.set_ylabel('velocity/mps')
        ax1.set_zlabel('magnitude/dB')
        ax1.set_title('2D-FFT')
        
        # 添加颜色条
        fig1.colorbar(surf, shrink=0.5, aspect=5)

        # 创建第二个图形：CFAR检测判决门限
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')

        # 绘制3D网格图
        X_cfar, Y_cfar = np.meshgrid(CFAR_MapRange_Dim, CFAR_MapDoppler_Dim)
        Z_cfar = 10 * np.log10(CFAR_Map.T)  # 转置以匹配MATLAB的mesh行为
        surf_cfar = ax2.plot_surface(X_cfar, Y_cfar, Z_cfar, cmap=cm.viridis, linewidth=0, antialiased=True)
        
        # 设置坐标轴标签和标题
        ax2.set_xlabel('range/m')
        ax2.set_ylabel('velocity/mps')
        ax2.set_zlabel('magnitude/dB')
        ax2.set_title('2D-CFAR detection judgment threshold')

        # 添加颜色条
        fig2.colorbar(surf_cfar, shrink=0.5, aspect=5)

        # 显示图形
        plt.tight_layout()
        plt.show()
        #-----------------------------------可视化------------------------------------------------------------
        
        # 截取与CFAR_MAP相同大小的区域
        # CRange_Doppler_Map = range_doppler_map[:CFAR_Map_r, :CFAR_Map_c]
        # 查找小于CFAR阈值的元素并置零
        comR, comC = np.where(DopplerFFT_out_result[RX_idx,:CFAR_Map_r,:CFAR_Map_c,frame_idx] < CFAR_Map)
        DopplerFFT_out_result[RX_idx,comR, comC,frame_idx] = 0
        # 查找非零元素的索引
        r1,c1 = np.nonzero(DopplerFFT_out_result[RX_idx,:CFAR_Map_r,:CFAR_Map_c,frame_idx])
        
        # CRange_Doppler_Map = CRange_Doppler_Map[:CFAR_Map_r,:]
        # 创建第一个图形：3D网格图（截取距离维度的第10到30个点）
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')

        # 截取距离维度的指定范围（注意Python索引从0开始，MATLAB从1开始）
        X = CFAR_MapRange_Dim  # MATLAB的10:30对应Python的9:30
        Y = CFAR_MapDoppler_Dim
        X, Y = np.meshgrid(X, Y)

        # 绘制3D网格图
        surf = ax1.plot_surface(X, Y,DopplerFFT_out_result[RX_idx,:CFAR_Map_r,:CFAR_Map_c,frame_idx].T, cmap=plt.cm.coolwarm, 
                            linewidth=0, antialiased=True)

        # 设置坐标轴标签和标题
        ax1.set_xlabel('Range/m')
        ax1.set_ylabel('Velocity/mps')
        ax1.set_zlabel('magnitude')
        ax1.set_title('CFAR-processed 3D mesh')

        # 添加颜色条
        fig1.colorbar(surf, shrink=0.5, aspect=5)

        # 创建第二个图形：2D图像
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)

        # 绘制2D图像
        im = ax2.imshow(np.abs(DopplerFFT_out_result[RX_idx,:CFAR_Map_r,:CFAR_Map_c,frame_idx].T), extent=[X.min(), X.max(), Y.min(), Y.max()],
                        aspect='auto', cmap=plt.cm.viridis, origin='lower')

        # 设置坐标轴标签和标题
        ax2.set_xlabel('Range/m')
        ax2.set_ylabel('Velocity/mps')
        ax2.set_title('2D image after CFAR processing')

        # 添加颜色条
        fig2.colorbar(im, shrink=0.5, aspect=5)

        # 显示图形
        plt.tight_layout()
        plt.show()
        return DopplerFFT_out_result[RX_idx,:CFAR_Map_r,:CFAR_Map_c,frame_idx]
    
    
    def Azimuth_FFT(self, DopplerFFT_out_result, param, frame_idx):
        azimuth_chs = np.r_[0:4,8:12]
        input = np.sum(DopplerFFT_out_result[:,:,:,frame_idx], axis=2)
        az_in = input[azimuth_chs,:]
        az_profile = np.fft.fftshift(np.fft.fft(az_in, n=param['angle_fftSize'], axis=0), axes=0)
        Range_Index = np.arange(param['range_fftSize']) * param['Range_Res']
        agl = np.degrees(np.arcsin(np.linspace(-1,1,param['angle_fftSize'])))
        
        plt.figure()
        plt.imshow(np.abs(az_profile), aspect='auto',
                   extent=[Range_Index[0], Range_Index[-1],agl[0], agl[-1]])
        plt.xlabel('Azimuth (°)')
        plt.ylabel('Range (m)')
        plt.title('Range-Azimuth Heatmap')
        plt.colorbar()
        plt.show()
        return None