import numpy as np
import copy
import warnings  # [新增] 引入 warnings 模块
from typing import Tuple, Optional, Dict
from scipy.signal import fftconvolve
# 引用现有模块
from litho_model.lithography_simulator import LithographySimulator
from litho_model.set_patterns import Mask

class PVBandComputer:
    """
    CPU 版本的光刻工艺窗口 (PV Band) 计算器。
    
    功能：
    计算 Dose (剂量) 和 Defocus (焦距) 变化下的光刻胶轮廓变化带 (PV Band)。
    
    """
    def __init__(self, 
                 simulator: LithographySimulator,
                 dose_margin: float = 0.05,      # 剂量容差 ±5%
                 dof_range_nm: float = 200.0,    # 焦深范围 (全宽)
                 dof_steps: int = 3,             # 离焦采样点数 (奇数: -d, 0, +d)
                 mode: str = 'full',                # 'dose_only', 'dof_only', 'full'
                 method: str = 'SOCS',):            # 'SOCS', 'Abbe'
        
        self.simulator = simulator
        self.dose_margin = dose_margin
        self.dof_range_nm = dof_range_nm
        self.dof_steps = dof_steps
        self.mode = mode
        self.method = method
        # 1. 提取基础参数
        self.th_nominal = simulator.params.resist.threshold
        self.alpha = simulator.params.resist.alpha
        self.na = simulator.params.optics.na
        self.refractive_index = simulator.params.optics.refractive_index
        self.wavelength = simulator.params.system.wavelength_nm
        
        # 2. 安全获取切趾设置
        self.enable_apodization = getattr(simulator.optics, 'enable_apodization', False)
        
        # 3. 预计算剂量对应的阈值变化
        self.th_fat = self.th_nominal * (1 - dose_margin) 
        self.th_slim = self.th_nominal * (1 + dose_margin)
        
        # 4. 预计算离焦采样点
        if dof_range_nm > 0 and dof_steps > 1:
            half_range = dof_range_nm / 2.0
            self.defocus_points = np.linspace(-half_range, half_range, dof_steps)
        else:
            self.defocus_points = np.array([0.0])
            
        print(f"PVBand 初始化完成:")
        print(f"  - 模式: {mode}")
        print(f"  - 离焦采样: {self.defocus_points} nm")
        print(f"  - 光学参数: NA={self.na}, n={self.refractive_index}")

    def _sigmoid_resist(self, aerial: np.ndarray, threshold: float) -> np.ndarray:
        val = -self.alpha * (aerial - threshold)
        val = np.clip(val, -50, 50) 
        return 1.0 / (1.0 + np.exp(val))
    
    def _update_global_extrema(self, 
                               current_wafer: np.ndarray, 
                               g_max: Optional[np.ndarray], # optional表示这个值可以为None
                               g_min: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """更新全局最大/最小值"""
        if g_max is None:
            return current_wafer.copy(), current_wafer.copy()
        else:
            return np.maximum(g_max, current_wafer), np.minimum(g_min, current_wafer)

    def _compute_z4_coefficient(self, defocus_nm: float) -> float:
        """
        计算物理离焦对应的 Zernike Z4 系数
        """
        z4_coeff = (self.na ** 2 * defocus_nm) / (4 * self.refractive_index * self.wavelength)
        return z4_coeff

    def _recompute_cache_at_defocus(self, defocus_nm: float) -> dict:
        """
        重新计算指定离焦下的光学核心
        """
        # 1. 准备像差系数
        temp_aberrations = self.simulator.params.optics.aberrations.copy()
        
        if abs(defocus_nm) > 1e-6:
            z4_val = self._compute_z4_coefficient(defocus_nm)
            temp_aberrations['z4'] = temp_aberrations.get('z4', 0.0) + z4_val
        
        # 2. 临时实例化 OpticalSystem (用于生成光瞳)
        from litho_model.set_optical_system import OpticalSystem
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*检测到高 NA.*")
            
            try:
                temp_optics = OpticalSystem(
                    na=self.na,
                    wavelength_nm=self.wavelength,
                    grid_size=self.simulator.mask.grid_size,
                    pitch=self.simulator.mask.pitch,
                    refractive_index=self.refractive_index,
                    aberrations=temp_aberrations,
                    enable_apodization=self.enable_apodization
                )
            except TypeError:
                # 兼容旧版本构造函数
                temp_optics = OpticalSystem(
                    na=self.na,
                    wavelength_nm=self.wavelength,
                    grid_size=self.simulator.mask.grid_size,
                    pitch=self.simulator.mask.pitch,
                    refractive_index=self.refractive_index,
                    aberrations=temp_aberrations
                )
        
        # optcache = {}
        # if not self.enable_apodization:
        # #     print("未启用光瞳切趾")
        #     if not optcache.get("printed", False):
        #         print("########### 未启用光瞳切趾 ############")
        #         optcache["printed"] = True
        # 3. 计算频域位移光瞳 (Shifted Pupils)
        pupil_func = temp_optics.pupil_function
        source = self.simulator.source
        
        sindy, sindx = np.nonzero(source.source_map > 1e-9)
        source_weights = source.source_map[sindy, sindx]
        
        pupil_k_step = temp_optics.frequency_coords[1] - temp_optics.frequency_coords[0]
        shifts_x = np.round(source.source_coords[sindx] / pupil_k_step).astype(int)
        shifts_y = np.round(source.source_coords[sindy] / pupil_k_step).astype(int)
        if self.method == "Abbe":
            shifted_pupils = []
            for sy, sx in zip(shifts_y, shifts_x):
                p_shift = np.roll(pupil_func, (sy, sx), axis=(0, 1))
                shifted_pupils.append(p_shift)
            
            return {
            "shifted_pupils": shifted_pupils,
            "source_weights": source_weights
            }
        if self.method == "SOCS":
            num_kernels = 50
            N = pupil_func.shape[0]
            M = len(source_weights) 
            #预组装A矩阵, 每一列是一个展平的sqrt(w_s) * P(k - ks)
            A = np.zeros((N * N, M),dtype= complex)
            for i in range(M):
                #对光瞳进行频域
                shifted_p = np.roll(pupil_func,(shifts_y[i], shifts_x[i]), axis=(0,1))
                #加入光源权重并拉平
                A[:,i] = np.ravel(np.sqrt(source_weights[i]) * shifted_p)
            #3.进行SVD分解,U左奇异值向量，形状为(N*N , M), 
            #S奇异值，形状为(M,), Vh右奇异值向量，形状为(M, M)
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            # 4. 提取特征值和核
            # 特征值 lambda = 奇异值 S 的平方
            lambdas = S**2
            
            # 提取前 K 个相干核
            socs_kernels_spatial = []
            actual_k = min(num_kernels, M)
            for i in range(actual_k):
                #频率核域，将U矩阵的一列提取并reshaoe为N*N
                kernel_freq = U[:, i].reshape((N, N))
                # 空间域核：由于光刻计算通常在空间域做卷积，所以这里做 IFFT
                kernel_spatial = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel_freq)))
                socs_kernels_spatial.append(kernel_spatial)

            return{
                "kernels":socs_kernels_spatial,
                "lambdas":lambdas[:actual_k],
                "norm_factor":np.sum(source_weights)
            }
    def _compute_aerial_image_cpu(self, mask_spatial: np.ndarray, opt_cache: dict) -> np.ndarray:
        """
        使用频域乘法计算空间像。
        """
        if self.method == "Abbe":
            shifted_pupils = opt_cache["shifted_pupils"]
            source_weights = opt_cache["source_weights"]
            
            # 1. 计算 Mask 频谱 (只需一次)
            d_fft = np.fft.ifftshift(mask_spatial)
            mask_spectrum = np.fft.fftshift(np.fft.fft2(d_fft))
            
            aerial_image = np.zeros_like(mask_spatial, dtype=np.float64)
            
            # 2. 遍历所有光源点进行非相干叠加
            for pupil, weight in zip(shifted_pupils, source_weights):
                E_freq = mask_spectrum * pupil
                E_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_freq)))
                aerial_image += weight * (np.abs(E_field) ** 2)
                
            # 3. 归一化
            total_intensity = np.sum(source_weights)
            if total_intensity > 1e-9:
                aerial_image /= total_intensity
                
            return aerial_image
        if self.method == "SOCS":
            #1. 取出计算好的kernals和lambas
            kernels = opt_cache["kernels"]
            lambdas = opt_cache["lambdas"]
            norm_factor = opt_cache["norm_factor"]
            aerial_image = np.zeros_like(mask_spatial, dtype=np.float64)
            
            #2. 遍历所有特征值进行非相干叠加
            for i in range(len(lambdas)):
                kernel = kernels[i]
                weight  = lambdas[i]
                E_field = fftconvolve(mask_spatial, kernel, mode='same')
                aerial_image += weight * (np.abs(E_field) ** 2)
            
            #3. 归一化
            aerial_image /= norm_factor

            return aerial_image

    def compute(self, mask_spatial: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        PV Band 计算
        """
        global_max, global_min = None, None
        
        # print(f"--- 开始 PV Band 计算 (模式: {self.mode}) ---")
        
        # 模式 1: Dose Only
        if self.mode == 'dose_only':
            temp_cache = self._recompute_cache_at_defocus(0.0)
            aerial = self._compute_aerial_image_cpu(mask_spatial, temp_cache)
            # 计算阈值高和阈值低的成像
            wafer_fat = self._sigmoid_resist(aerial, self.th_fat)
            wafer_slim = self._sigmoid_resist(aerial, self.th_slim)
            pv_map = np.maximum(0, wafer_fat - wafer_slim)
            
        # 模式 2 & 3: 涉及离焦
        else:
            thresholds = [self.th_fat, self.th_nominal, self.th_slim] if self.mode == 'full' else [self.th_nominal]
            
            for defocus in self.defocus_points:
                # 1. 动态计算光学核
                temp_cache = self._recompute_cache_at_defocus(defocus)
                
                # 2. 计算空间像
                aerial = self._compute_aerial_image_cpu(mask_spatial, temp_cache)
                
                # 3. 遍历剂量阈值更新极值
                for th in thresholds:
                    wafer = self._sigmoid_resist(aerial, th)
                    global_max, global_min = self._update_global_extrema(wafer, global_max, global_min)
            
            pv_map = global_max - global_min

        # 计算PVband Loss = pv_loss
        
        # method 1 全图平均值
        # pv_loss = np.mean(pv_map)
        
        # method 2  L2 范数
        # pv_loss = np.linalg.norm(pv_map)
        
        # method 3 求和 ,衡量总失效面积
        pv_loss = np.sum(pv_map)
        
        # method 4 只统计边缘区域 ,衡量-总失效面积的平均值，也可以改为求和/二范数/...
        # valid_pixels = pv_map[pv_map > 1e-6]
        # if valid_pixels.size > 0:
        #     pv_loss = np.mean(valid_pixels)
        # else:
        #     pv_loss = 0.0
        
        # print(f"--- PV Band 计算完成. Loss: {pv_loss:.6f} ---")
        
        return pv_loss, pv_map