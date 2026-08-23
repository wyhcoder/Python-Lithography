"""
PV Loss 显式梯度计算模块 (CPU版本)

物理模型：
- 支持 Dose ± margin 和 Defocus ± range 的工艺窗口

"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from litho_model.lithography_simulator import LithographySimulator
from litho_model.set_optical_system import OpticalSystem


@dataclass
class ProcessCondition:
    """工艺条件数据类"""
    defocus_nm: float      # 离焦量 (nm)
    threshold: float       # 光刻胶阈值 (对应 Dose 变化)
    label: str             # 标签 (用于调试)
    
    
class OpticalCache:
    """
    光学计算缓存管理器
    
    功能：
    1. 缓存 Mask FFT，避免重复计算
    2. 缓存各离焦条件下的位移光瞳
    3. 缓存空间像结果，供 PE/PV Loss 共享
    """
    def __init__(self):
        self.mask_spectrum: Optional[np.ndarray] = None          # Mask 频谱
        self.mask_hash: Optional[int] = None                     # Mask 哈希值 (用于检测变化)
        self.pupil_cache: Dict[float, Dict] = {}                 # {defocus: {shifted_pupils, weights}}
        self.aerial_cache: Dict[Tuple[float, float], np.ndarray] = {}  # {(defocus, th): aerial}
        
    def is_mask_valid(self, mask: np.ndarray) -> bool:
        """检查缓存的 Mask 是否仍然有效"""
        current_hash = hash(mask.tobytes())
        return self.mask_hash == current_hash
    
    def update_mask_spectrum(self, mask: np.ndarray):
        """更新 Mask 频谱缓存"""
        self.mask_hash = hash(mask.tobytes())
        d_fft = np.fft.ifftshift(mask)
        self.mask_spectrum = np.fft.fftshift(np.fft.fft2(d_fft))
        # Mask 变化后清空依赖缓存
        self.aerial_cache.clear()
        
    def clear(self):
        """清空所有缓存"""
        self.mask_spectrum = None
        self.mask_hash = None
        self.pupil_cache.clear()
        self.aerial_cache.clear()


class PVLossWithGradient:
    """
    带显式梯度计算的 PV Loss (CPU 版本)
    
    使用方法：
    ```python
    pv_loss = PVLossWithGradient(simulator)
    
    # 方法1: PV
    loss, grad, pv_map = pv_loss.compute_with_gradient(mask)
    
    # 方法2: PV + PE Loss
    cache = OpticalCache()
    aerial_nominal = pv_loss.compute_aerial_image(mask, defocus=0, cache=cache)
    loss_pe = compute_pe_loss(aerial_nominal, target)
    loss_pv, grad_pv = pv_loss.compute_from_cache(mask, cache)
    ```
    """
    
    def __init__(self, 
                 simulator: LithographySimulator,
                 dose_margin: float = 0.05,
                 dof_range_nm: float = 200.0,
                 dof_steps: int = 3,
                 mode: str = 'full',
                 memory_efficient: bool = False):
        """
        Args:
            simulator: 光刻仿真器实例
            dose_margin: 剂量容差 (±5% → 0.05)
            dof_range_nm: 焦深范围全宽 (nm)
            dof_steps: 离焦采样点数 (奇数)
            mode: 'dose_only', 'dof_only', 'full'
        """
        self.simulator = simulator
        self.dose_margin = dose_margin
        self.dof_range_nm = dof_range_nm
        self.dof_steps = dof_steps
        self.mode = mode
        self.memory_efficient = memory_efficient
        
        # 提取物理参数
        self.optcache = {}
        self.th_nominal = simulator.params.resist.threshold
        self.alpha = simulator.params.resist.alpha  # Sigmoid 陡度
        self.na = simulator.params.optics.na
        self.refractive_index = simulator.params.optics.refractive_index
        self.wavelength = simulator.params.system.wavelength_nm
        self.enable_apodization = getattr(simulator.optics, 'enable_apodization', False)
        # if not self.enable_apodization:
        # #     print("未启用光瞳切趾")
        #     if not self.optcache.get("printed", False):
        #         print("########### 未启用光瞳切趾 ############")
        #         self.optcache["printed"] = True
        # 预计算阈值
        self.th_fat = self.th_nominal * (1 - dose_margin)   # 低阈值 → 易曝光 → 胖轮廓
        self.th_slim = self.th_nominal * (1 + dose_margin)  # 高阈值 → 难曝光 → 瘦轮廓
        
        # 预计算离焦点
        if dof_range_nm > 0 and dof_steps > 1:
            half_range = dof_range_nm / 2.0
            self.defocus_points = np.linspace(-half_range, half_range, dof_steps)
        else:
            self.defocus_points = np.array([0.0])
            
        # 生成工艺条件列表
        self.process_conditions = self._generate_process_conditions()
        
        # 内部缓存
        self._cache = OpticalCache()
        
        print(f"PVLossWithGradient 初始化完成:")
        print(f"  - 模式: {mode}")
        print(f"  - 工艺条件数: {len(self.process_conditions)}")
        print(f"  - 离焦点: {self.defocus_points} nm")
        
    def _generate_process_conditions(self) -> List[ProcessCondition]:
        """生成所有工艺条件组合"""
        conditions = []
        
        if self.mode == 'dose_only':
            # 仅 Dose 变化，Defocus = 0
            conditions.append(ProcessCondition(0.0, self.th_fat, "D+/F0"))
            conditions.append(ProcessCondition(0.0, self.th_slim, "D-/F0"))
            
        elif self.mode == 'dof_only':
            # 仅 Defocus 变化，Dose = nominal
            for df in self.defocus_points:
                conditions.append(ProcessCondition(df, self.th_nominal, f"D0/F{df:+.0f}"))
                
        else:  # 'full'
            # Dose × Defocus 全组合
            thresholds = [
                (self.th_fat, "D+"),
                (self.th_nominal, "D0"),
                (self.th_slim, "D-")
            ]
            for df in self.defocus_points:
                for th, th_label in thresholds:
                    conditions.append(ProcessCondition(df, th, f"{th_label}/F{df:+.0f}"))
                    
        return conditions
    
    def _compute_z4_coefficient(self, defocus_nm: float) -> float:
        """
        计算 Zernike Z4 离焦系数 (High-NA 修正版)
        
        公式: Z4 = (NA² × Δz) / (4 × n × λ)
        """
        return (self.na ** 2 * defocus_nm) / (4 * self.refractive_index * self.wavelength)
    
    def _get_or_create_pupil_cache(self, defocus_nm: float) -> Dict:
        """获取或创建指定离焦的光瞳缓存，将光瞳中的变量保存在字典中"""
        # 检查缓存
        if defocus_nm in self._cache.pupil_cache:
            return self._cache.pupil_cache[defocus_nm]
        
        # 创建新的光学系统
        temp_aberrations = self.simulator.params.optics.aberrations.copy()
        if abs(defocus_nm) > 1e-6:
            z4_val = self._compute_z4_coefficient(defocus_nm)
            temp_aberrations['z4'] = temp_aberrations.get('z4', 0.0) + z4_val
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*检测到高 NA.*")
            temp_optics = OpticalSystem(
                na=self.na,
                wavelength_nm=self.wavelength,
                grid_size=self.simulator.mask.grid_size,
                pitch=self.simulator.mask.pitch,
                refractive_index=self.refractive_index,
                aberrations=temp_aberrations,
                enable_apodization=self.enable_apodization
            )
        
        # 计算位移光瞳
        pupil_func = temp_optics.pupil_function
        source = self.simulator.source
        
        sindy, sindx = np.nonzero(source.source_map > 1e-9)
        source_weights = source.source_map[sindy, sindx]
        
        pupil_k_step = temp_optics.frequency_coords[1] - temp_optics.frequency_coords[0]
        shifts_x = np.round(source.source_coords[sindx] / pupil_k_step).astype(int)
        shifts_y = np.round(source.source_coords[sindy] / pupil_k_step).astype(int)
        
        shifted_pupils = []
        for sy, sx in zip(shifts_y, shifts_x):
            p_shift = np.roll(pupil_func, (sy, sx), axis=(0, 1))
            shifted_pupils.append(p_shift)
        
        cache_entry = {
            "shifted_pupils": shifted_pupils,
            "source_weights": source_weights,
            "pupil_function": pupil_func,
            "shifts": list(zip(shifts_y, shifts_x))
        }
        
        self._cache.pupil_cache[defocus_nm] = cache_entry
        return cache_entry
    
    def _compute_aerial_image_with_fields(self, 
                                          mask_spectrum: np.ndarray,
                                          pupil_cache: Dict) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        计算空间像，同时返回各光源点的电场分布 (用于梯度计算)
        
        Returns:
            aerial_image: 空间像强度 [H, W]
            E_fields: 各光源点的电场 [N_source, H, W] (复数)
        """
        shifted_pupils = pupil_cache["shifted_pupils"]
        source_weights = pupil_cache["source_weights"]
        
        H, W = mask_spectrum.shape
        aerial_image = np.zeros((H, W), dtype=np.float64)
        E_fields = []
        
        for pupil, weight in zip(shifted_pupils, source_weights):
            E_freq = mask_spectrum * pupil
            E_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_freq)))
            E_fields.append(E_field)
            aerial_image += weight * (np.abs(E_field) ** 2)
        
        # 归一化
        total_weight = np.sum(source_weights)
        if total_weight > 1e-9:
            aerial_image /= total_weight
            
        return aerial_image, E_fields
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid 函数 (带溢出保护)"""
        val = np.clip(-self.alpha * x, -50, 50)
        return 1.0 / (1.0 + np.exp(val))
    
    def _sigmoid_derivative(self, sigmoid_output: np.ndarray) -> np.ndarray:
        """Sigmoid 导数: σ'(x) = α × σ(x) × (1 - σ(x))"""
        return self.alpha * sigmoid_output * (1.0 - sigmoid_output)
    
    def _compute_dI_dM(self, 
                       E_fields: List[np.ndarray],
                       source_weights: np.ndarray,
                       pupil_cache: Dict) -> np.ndarray:
        """
        计算空间像对 Mask 的梯度 (Hopkins 模型)
        
        公式:
        ∂I/∂M = 2 × Re[ Σ_s w_s × E_s* × F⁻¹{P_s × F{·}} ]
        
        注意：这是一个线性算子，返回的是 "梯度模板"
        实际使用时需要与上游梯度相乘
        """
        shifted_pupils = pupil_cache["shifted_pupils"]
        total_weight = np.sum(source_weights)
        
        H, W = E_fields[0].shape
        grad_template = np.zeros((H, W), dtype=np.complex128)
        
        for E_field, pupil, weight in zip(E_fields, shifted_pupils, source_weights):
            # E* × P (共轭乘以光瞳)
            # 注意：这里需要计算 ∂|E|²/∂M = 2 Re(E* × ∂E/∂M)
            # 而 ∂E/∂M 通过 FFT 和光瞳滤波实现
            # 简化: 返回 E* × P 作为模板
            E_conj = np.conj(E_field)
            grad_template += (weight / total_weight) * E_conj
            
        return grad_template
    
    def compute_aerial_image(self, 
                             mask_spatial: np.ndarray,
                             defocus_nm: float = 0.0,
                             cache: Optional[OpticalCache] = None) -> np.ndarray:
        """
        计算指定离焦下的空间像 (支持外部缓存共享)
        
        Args:
            mask_spatial: 掩模空间分布 [H, W]
            defocus_nm: 离焦量 (nm)
            cache: 外部缓存对象 (可选)
            
        Returns:
            aerial_image: 空间像 [H, W]
        """
        use_cache = cache if cache is not None else self._cache
        
        # 更新 Mask 频谱缓存
        if not use_cache.is_mask_valid(mask_spatial):
            use_cache.update_mask_spectrum(mask_spatial)
        
        # 获取光瞳缓存
        pupil_cache = self._get_or_create_pupil_cache(defocus_nm)
        
        # 计算空间像
        aerial, _ = self._compute_aerial_image_with_fields(
            use_cache.mask_spectrum, pupil_cache
        )
        
        return aerial
    
    
    def compute_with_gradient(self, 
                              mask_spatial: np.ndarray,
                              return_details: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        [优化版] 计算 PV Loss 及其显式梯度
        
        优化策略：
        1. 按 Defocus 分组聚合梯度，大幅减少 FFT/IFFT 次数
        2. 同一 Defocus 下的多个 Dose 条件共享光学场计算
        3. 先累加 ∂L/∂I，再执行一次光学反传
        
        性能提升：
        - 3 Dose × 3 Defocus: 从 9 次反传 → 3 次 (3× 加速)
        - N Dose × M Defocus: 从 N×M 次 → M 次 (N× 加速)
        
        Args:
            mask_spatial: 掩模空间分布 [H, W]
            return_details: 是否返回详细信息 (保留接口兼容性)
            
        Returns:
            pv_loss: PV Loss 值
            gradient: ∂L_PV/∂M [H, W]
            pv_map: PV Band 分布图 [H, W]
        """
        # 确保 Mask 频谱缓存有效
        if not self._cache.is_mask_valid(mask_spatial):
            self._cache.update_mask_spectrum(mask_spatial)
        
        H, W = mask_spatial.shape
        
        # === 阶段 1: 前向计算 (按 Defocus 分组以节省内存/计算) ===
        # 数据结构: { defocus_nm: { 'aerial': array, 'E_fields': list, 'pupil_cache': dict, 'condition_indices': [k1, k2...] } }
        optics_group_data = {} 
        
        # 预分配 wafer 数组，避免 list append
        num_conds = len(self.process_conditions)
        all_wafers = np.zeros((num_conds, H, W), dtype=np.float64)
        
        for k, cond in enumerate(self.process_conditions):
            df = cond.defocus_nm
            
            # 1. 获取光学场 (如果该 Defocus 还没算过)
            if df not in optics_group_data:
                pupil_cache = self._get_or_create_pupil_cache(df)
                aerial, E_fields = self._compute_aerial_image_with_fields(
                    self._cache.mask_spectrum, pupil_cache
                )
                optics_group_data[df] = {
                    'aerial': aerial,
                    'E_fields': E_fields if not self.memory_efficient else None,
                    'pupil_cache': pupil_cache,
                    'condition_indices': []
                }
            
            # 记录该条件属于哪个 Defocus 组
            optics_group_data[df]['condition_indices'].append(k)
            
            # 2. 计算光刻胶响应 (复用已计算的 aerial image)
            aerial = optics_group_data[df]['aerial']
            wafer = self._sigmoid(aerial - cond.threshold)
            all_wafers[k] = wafer
            
        # === 阶段 2: 计算 PV Band ===
        W_max = np.max(all_wafers, axis=0)  # [H, W]
        W_min = np.min(all_wafers, axis=0)  # [H, W]
        pv_map = np.maximum(0, W_max - W_min)
        pv_loss = np.sum(pv_map)
        
        # === 阶段 3: 计算显式梯度 (按 Defocus 分组聚合优化) ===
        idx_max = np.argmax(all_wafers, axis=0)  # [H, W] 每个像素的 max 来源
        idx_min = np.argmin(all_wafers, axis=0)  # [H, W] 每个像素的 min 来源
        
        total_gradient = np.zeros((H, W), dtype=np.float64)
        
        for df, group_data in optics_group_data.items():
            
            # 初始化该光学条件下的总空间像梯度 dL/dI_optical
            dL_dI_optical_accum = np.zeros((H, W), dtype=np.float64)
            is_involved = False  # 标记该 Defocus 组是否有梯度贡献
            
            # 遍历该 Defocus 下的所有 Dose 条件 (例如 Fat, Nominal, Slim)
            for k in group_data['condition_indices']:
                cond = self.process_conditions[k]
                
                max_mask = (idx_max == k)  #找到最大值所在的区域
                min_mask = (idx_min == k)
                
                if not (np.any(max_mask) or np.any(min_mask)):
                    continue
                
                is_involved = True
                wafer_k = all_wafers[k]
                sigmoid_deriv = self._sigmoid_derivative(wafer_k)
                
                if np.any(max_mask):
                    dL_dI_optical_accum += sigmoid_deriv * max_mask.astype(np.float64)
                
                if np.any(min_mask):
                    dL_dI_optical_accum -= sigmoid_deriv * min_mask.astype(np.float64)
            
            if is_involved:
                if self.memory_efficient:
                    _, E_fields = self._compute_aerial_image_with_fields(
                        self._cache.mask_spectrum, group_data['pupil_cache']
                    )
                else:
                    E_fields = group_data['E_fields']
                
                dI_dM_optical = self._backprop_aerial_to_mask(
                    dL_dI_optical_accum,
                    E_fields,
                    group_data['pupil_cache']
                )
                total_gradient += dI_dM_optical
                
        return pv_loss, total_gradient, pv_map
    
    def _backprop_aerial_to_mask(self,
                                  dL_dI: np.ndarray,
                                  E_fields: List[np.ndarray],
                                  pupil_cache: Dict) -> np.ndarray:
        """
        反向传播：从 ∂L/∂I 计算 ∂L/∂M
        
        Hopkins 模型:
        I = Σ_s w_s |E_s|²
        E_s = F⁻¹{ M̃ × P_s }
        M̃ = F{M}
        
        反向:
        ∂L/∂M = F⁻¹{ Σ_s w_s × P_s* × F{ 2 Re(E_s* × ∂L/∂I × E_s) } }
        
        简化（因为 I 是实数）:
        ∂L/∂M = 2 × Re{ F⁻¹{ Σ_s w_s × P_s* × F{ E_s* × ∂L/∂I } } }
        """
        shifted_pupils = pupil_cache["shifted_pupils"]
        source_weights = pupil_cache["source_weights"]
        total_weight = np.sum(source_weights)
        
        H, W = dL_dI.shape
        grad_freq_sum = np.zeros((H, W), dtype=np.complex128)
        
        for E_field, pupil, weight in zip(E_fields, shifted_pupils, source_weights):
            # E* × (∂L/∂I) — 注意 ∂L/∂I 是实数
            adjoint_field = np.conj(E_field) * dL_dI
            
            # FFT 变换
            adjoint_freq = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(adjoint_field)))
            
            # 乘以共轭光瞳
            grad_freq_sum += (weight / total_weight) * np.conj(pupil) * adjoint_freq
        
        # IFFT 得到空间域梯度
        grad_spatial = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grad_freq_sum)))
        
        # 取实部并乘以 2 (来自 |E|² 的导数)
        return 2.0 * np.real(grad_spatial)
    
    def compute_combined_loss_and_gradient(self,
                                           mask_spatial: np.ndarray,
                                           target: np.ndarray,
                                           pv_weight: float = 0.1) -> Dict:
        """
        组合计算 PE Loss + PV Loss，最大化共享计算
        
        优化策略：
        1. Mask FFT 只计算一次，PE/PV 共享
        2. Nominal 条件的 aerial/E_fields 被 PE Loss 和 PV Loss 共享
        3. 按 Defocus 分组聚合 PV 梯度
        
        Args:
            mask_spatial: 掩模 [H, W]
            target: 目标图案 [H, W]
            pv_weight: PV Loss 权重
            
        Returns:
            dict: {
                'loss_pe': float,
                'loss_pv': float,
                'loss_total': float,
                'grad_pe': ndarray,
                'grad_pv': ndarray,
                'grad_total': ndarray,
                'aerial_nominal': ndarray,
                'pv_map': ndarray
            }
        """
        # === 0. 确保 Mask 频谱缓存有效 ===
        if not self._cache.is_mask_valid(mask_spatial):
            self._cache.update_mask_spectrum(mask_spatial)
        
        H, W = mask_spatial.shape
        
        # === 1. 前向计算 (按 Defocus 分组，与 PV 计算共享) ===
        optics_group_data = {}
        num_conds = len(self.process_conditions)
        all_wafers = np.zeros((num_conds, H, W), dtype=np.float64)
        
        for k, cond in enumerate(self.process_conditions):
            df = cond.defocus_nm
            
            if df not in optics_group_data:
                pupil_cache = self._get_or_create_pupil_cache(df)
                aerial, E_fields = self._compute_aerial_image_with_fields(
                    self._cache.mask_spectrum, pupil_cache
                )
                optics_group_data[df] = {
                    'aerial': aerial,
                    'E_fields': E_fields if not self.memory_efficient else None,
                    'pupil_cache': pupil_cache,
                    'condition_indices': []
                }
            
            optics_group_data[df]['condition_indices'].append(k)
            aerial = optics_group_data[df]['aerial']
            wafer = self._sigmoid(aerial - cond.threshold)
            all_wafers[k] = wafer
        
        # === 2. 计算 PE Loss (使用 Nominal 条件) ===
        # 获取 Nominal 光学数据
        nominal_group = optics_group_data.get(0.0)
        if nominal_group is None:
            # 如果没有 defocus=0 的条件，单独计算
            pupil_cache_nominal = self._get_or_create_pupil_cache(0.0)
            aerial_nominal, E_fields_nominal = self._compute_aerial_image_with_fields(
                self._cache.mask_spectrum, pupil_cache_nominal
            )
        else:
            aerial_nominal = nominal_group['aerial']
            if self.memory_efficient:
                _, E_fields_nominal = self._compute_aerial_image_with_fields(
                    self._cache.mask_spectrum, nominal_group['pupil_cache']
                )
            else:
                E_fields_nominal = nominal_group['E_fields']
            pupil_cache_nominal = nominal_group['pupil_cache']
        
        wafer_nominal = self._sigmoid(aerial_nominal - self.th_nominal)
        error = wafer_nominal - target
        loss_pe = np.sum(error ** 2)
        
        # PE 梯度: ∂L_PE/∂M = 2(W - T) × σ'(I - th) × ∂I/∂M
        dL_dW_pe = 2.0 * error
        sigmoid_deriv_nominal = self._sigmoid_derivative(wafer_nominal)
        dL_dI_pe = dL_dW_pe * sigmoid_deriv_nominal
        grad_pe = self._backprop_aerial_to_mask(dL_dI_pe, E_fields_nominal, pupil_cache_nominal)
        
        # === 3. 计算 PV Loss (按 Defocus 分组优化) ===
        W_max = np.max(all_wafers, axis=0)
        W_min = np.min(all_wafers, axis=0)
        pv_map = np.maximum(0, W_max - W_min)
        loss_pv = np.sum(pv_map)
        
        # PV 梯度 (按 Defocus 分组聚合)
        idx_max = np.argmax(all_wafers, axis=0)
        idx_min = np.argmin(all_wafers, axis=0)
        
        grad_pv = np.zeros((H, W), dtype=np.float64)
        
        for df, group_data in optics_group_data.items():
            dL_dI_accum = np.zeros((H, W), dtype=np.float64)
            is_involved = False
            
            for k in group_data['condition_indices']:
                max_mask = (idx_max == k)
                min_mask = (idx_min == k)
                
                if not (np.any(max_mask) or np.any(min_mask)):
                    continue
                
                is_involved = True
                wafer_k = all_wafers[k]
                sigmoid_deriv = self._sigmoid_derivative(wafer_k)
                
                if np.any(max_mask):
                    dL_dI_accum += sigmoid_deriv * max_mask.astype(np.float64)
                if np.any(min_mask):
                    dL_dI_accum -= sigmoid_deriv * min_mask.astype(np.float64)
            
            if is_involved:
                if self.memory_efficient:
                    _, E_fields = self._compute_aerial_image_with_fields(
                        self._cache.mask_spectrum, group_data['pupil_cache']
                    )
                else:
                    E_fields = group_data['E_fields']
                
                dI_dM = self._backprop_aerial_to_mask(
                    dL_dI_accum, E_fields, group_data['pupil_cache']
                )
                grad_pv += dI_dM
        
        # === 4. 组合 ===
        loss_total = loss_pe + pv_weight * loss_pv
        grad_total = grad_pe + pv_weight * grad_pv
        
        return {
            'loss_pe': loss_pe,
            'loss_pv': loss_pv,
            'loss_total': loss_total,
            'grad_pe': grad_pe,
            'grad_pv': grad_pv,
            'grad_total': grad_total,
            'aerial_nominal': aerial_nominal,
            'pv_map': pv_map
        }
    
    def clear_cache(self):
        """清空所有缓存"""
        self._cache.clear()
        
        
        
# ============================================================================
# 优化器集成示例（借鉴梯度调用格式）
# ============================================================================

class CombinedOptimizer:
    """
    组合优化器示例：PE + PV Loss
    
    演示如何在优化循环中高效使用 PVLossWithGradient
    """
    
    def __init__(self, 
                 simulator: LithographySimulator,
                 pv_weight: float = 0.1,
                 learning_rate: float = 0.01):
        """
        Args:
            simulator: 光刻仿真器
            pv_weight: PV Loss 权重
            learning_rate: 学习率
        """
        self.pv_loss_module = PVLossWithGradient(
            simulator,
            dose_margin=0.05,
            dof_range_nm=200.0,
            dof_steps=3,
            mode='full'
        )
        self.pv_weight = pv_weight
        self.lr = learning_rate
        
    def optimize(self,
                 initial_mask: np.ndarray,
                 target: np.ndarray,
                 max_iterations: int = 100,
                 verbose: bool = True) -> np.ndarray:
        """
        执行优化
        
        Args:
            initial_mask: 初始掩模 [H, W]
            target: 目标图案 [H, W]
            max_iterations: 最大迭代次数
            verbose: 是否打印进度
            
        Returns:
            optimized_mask: 优化后的掩模 [H, W]
        """
        mask = initial_mask.copy().astype(np.float64)
        
        if verbose:
            print(f"{'Iter':<6} | {'Total':<12} | {'PE':<12} | {'PV':<12}")
            print("-" * 50)
        
        best_loss = float('inf')
        best_mask = mask.copy()
        
        for i in range(max_iterations):
            # 计算组合 Loss 和梯度 (共享计算)
            result = self.pv_loss_module.compute_combined_loss_and_gradient(
                mask, target, self.pv_weight
            )
            
            # 梯度下降更新
            mask = mask - self.lr * result['grad_total']
            
            # 投影到 [0, 1]
            mask = np.clip(mask, 0, 1)
            
            # 记录最佳
            if result['loss_total'] < best_loss:
                best_loss = result['loss_total']
                best_mask = mask.copy()
            
            # 打印进度
            if verbose and (i + 1) % 10 == 0:
                print(f"{i+1:<6} | {result['loss_total']:<12.2f} | "
                      f"{result['loss_pe']:<12.2f} | {result['loss_pv']:<12.2f}")
        
        return best_mask