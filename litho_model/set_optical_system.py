# 第一版
# import numpy as np
# from typing import Dict, Callable

# class ZernikeGenerator:
#     """
#     Zernike多项式生成器
#     """
#     def __init__(self, x: np.ndarray, y: np.ndarray):
#         self.x = x
#         self.y = y
#         self._cache: Dict[int, np.ndarray] = {}
#         self._polynomial_map: Dict[int, Callable[[], np.ndarray]] = {
#             2: lambda: self.x,
#             3: lambda: self.y,
#             4: lambda: 2*(self.x**2 + self.y**2) - 1,
#             5: lambda: self.x**2 - self.y**2,
#             6: lambda: 2*self.x*self.y,
#             7: lambda: (3*(self.x**2 + self.y**2) - 2) * self.x,
#             8: lambda: (3*(self.x**2 + self.y**2) - 2) * self.y,
#             9: lambda: 6*((self.x**2 + self.y**2 - 1)*(self.x**2 + self.y**2)) + 1,
#             10: lambda: self.x**3 - 3*self.x*self.y**2,
#             11: lambda: 3*self.y*self.x**2 - self.y**3,
#             12: lambda: (self.x**2 - self.y**2) * (4*(self.x**2 + self.y**2) - 3),
#             13: lambda: 2 * (4*(self.x**2 + self.y**2) - 3) * self.x*self.y,
#             14: lambda: (10*(self.x**2 + self.y**2)**2 - 12*(self.x**2 + self.y**2) + 3) * self.x,
#             15: lambda: (10*(self.x**2 + self.y**2)**2 - 12*(self.x**2 + self.y**2) + 3) * self.y,
#             16: lambda: 2*((self.x**2 + self.y**2)*(10*(self.x**2 + self.y**2)**2 - 15*(self.x**2 + self.y**2) + 6)) - 1,
#             17: lambda: (self.x**2 - self.y**2)**2 - (2*self.x*self.y)**2,
#             18: lambda: 4 * (self.x**2 - self.y**2) * self.x * self.y,
#             19: lambda: 5*self.x**5 - 10*self.x**3*self.y**2 - 15*self.x*self.y**4 - 4*self.x**3 + 12*self.x*self.y**2,
#             20: lambda: 15*self.x**4*self.y + 10*self.y**3*self.x**2 - 5*self.y**5 - 12*self.y*self.x**2 + 4*self.y**3,
#             21: lambda: (self.x**2 - self.y**2) * (15*(self.x**2 + self.y**2)**2 - 20*(self.x**2 + self.y**2) + 6),
#             22: lambda: 2 * (15*(self.x**2 + self.y**2)**2 - 20*(self.x**2 + self.y**2) + 6) * self.x * self.y,
#             23: lambda: (35*(self.x**2+self.y**2)**3 - 60*(self.x**2+self.y**2)**2 + 30*(self.x**2+self.y**2) - 4)*self.x,
#             24: lambda: (35*(self.x**2+self.y**2)**3 - 60*(self.x**2+self.y**2)**2 + 30*(self.x**2+self.y**2) - 4)*self.y,
#             25: lambda: 10*((self.x**2+self.y**2-1)*(self.x**2+self.y**2)*(7*(self.x**2+self.y**2)**2-7*(self.x**2+self.y**2)+2))+1,
#             26: lambda: self.x**5 - 10*self.x**3*self.y**2 + 5*self.x*self.y**4,
#             27: lambda: 5*self.x**4*self.y - 10*self.y**3*self.x**2 + self.y**5,
#             28: lambda: (6*(self.x**2+self.y**2)-5)*( (self.x**2-self.y**2)**2 - (2*self.x*self.y)**2 ),
#             29: lambda: 4*(self.x**2-self.y**2)*(6*(self.x**2+self.y**2)-5)*self.x*self.y,
#             30: lambda: (21*self.x**7-21*self.x**5*self.y**2-105*self.x**3*self.y**4-63*self.x*self.y**6-30*self.x**5+60*self.x**3*self.y**2+90*self.x*self.y**4+10*self.x**3-30*self.x*self.y**2),
#             31: lambda: (63*self.y*self.x**6+105*self.y**3*self.x**4+21*self.y**5*self.x**2-21*self.y**7-90*self.x**4*self.y-60*self.y**3*self.x**2+30*self.y**5+30*self.y*self.x**2-10*self.y**3),
#             32: lambda: (self.x**2-self.y**2)*(56*(self.x**2+self.y**2)**3-105*(self.x**2+self.y**2)**2+60*(self.x**2+self.y**2)-10),
#             33: lambda: 2*(56*(self.x**2+self.y**2)**3-105*(self.x**2+self.y**2)**2+60*(self.x**2+self.y**2)-10)*self.x*self.y,
#             34: lambda: (126*(self.x**2+self.y**2)**4-280*(self.x**2+self.y**2)**3+210*(self.x**2+self.y**2)**2-60*(self.x**2+self.y**2)+5)*self.x,
#             35: lambda: (126*(self.x**2+self.y**2)**4-280*(self.x**2+self.y**2)**3+210*(self.x**2+self.y**2)**2-60*(self.x**2+self.y**2)+5)*self.y,
#             36: lambda: 2*((self.x**2+self.y**2)*(126*(self.x**2+self.y**2)**4-315*(self.x**2+self.y**2)**3+280*(self.x**2+self.y**2)**2-105*(self.x**2+self.y**2)+15))-1,
#             37: lambda: 42*((self.x**2+self.y**2)*(self.x**2+self.y**2-1)*(22*(self.x**2+self.y**2)**4-44*(self.x**2+self.y**2)**3+31*(self.x**2+self.y**2)**2-9*(self.x**2+self.y**2)+1))+1,
#         }

#     def get(self, n: int) -> np.ndarray:
#         if n in self._cache:
#             return self._cache[n]
#         calculator = self._polynomial_map.get(n, lambda: np.zeros_like(self.x))
#         result = calculator()
#         self._cache[n] = result
#         return result

# class OpticalSystem:
#     """
#     负责模拟投影物镜的光学特性，核心是计算光瞳函数。
#     """
#     def __init__(self, 
#                  na: float, 
#                  wavelength_nm: float,
#                  grid_size: int,
#                  pitch: float,
#                  aberrations: Dict[str, float] = None):
#         self.na = na
#         self.wavelength_nm = wavelength_nm
#         self.grid_size = grid_size
#         self.pitch = pitch
#         self.aberrations = aberrations if aberrations is not None else {}
#         self.specimen, self.frequency_coords, self.pupil_function = self._compute_pupil()

#     def _compute_pupil(self) -> tuple[np.ndarray, np.ndarray]:
#         """
#         根据系统参数计算光瞳函数。
#         """
#         norm_pitch = self.pitch * self.na / self.wavelength_nm
#         specimen = 1 / norm_pitch
#         ff_coord = np.linspace(-(self.grid_size-1)/2, (self.grid_size-1)/2, self.grid_size) * specimen
#         normalized_ffx, normalized_ffy = np.meshgrid(ff_coord, ff_coord)
        
#         # 实例化Zernike生成器
#         zernike_gen = ZernikeGenerator(normalized_ffx, normalized_ffy)

#         # 计算总的波前像差函数 (Wavefront Aberration, W)
#         W = np.zeros_like(normalized_ffx)
#         for z_index_str, coefficient in self.aberrations.items():
#             try:
#                 z_index = int(z_index_str.replace('z', ''))
#                 W += coefficient * zernike_gen.get(z_index)
#             except (ValueError, TypeError):
#                 print(f"警告: 无法解析像差索引 '{z_index_str}'。已跳过。")

#         # 生成最终的光瞳函数 P = Aperture * exp(j * 2 * pi * W)
#         r_sq = normalized_ffx**2 + normalized_ffy**2
#         aperture_mask = (r_sq <= 1).astype(float)
        
#         phase = np.exp(1j * 2 * np.pi * W)
#         pupil_pattern = aperture_mask * phase
        
#         return specimen, ff_coord, pupil_pattern


# 第二版

import numpy as np
import warnings
import math
from typing import Dict, Callable, Optional
from enum import Enum
from utils_model.project_paths import DIAGNOSTICS_OUTPUT_DIR


class ZernikeConvention(Enum):
    """
    Zernike 多项式索引约定枚举。
    
    不同软件和文献使用不同的 Zernike 索引约定，主要有：
    - NOLL: 天文学和大气光学领域常用，按径向阶次排序（本代码内部标准）
    - FRINGE: 光学工业（如 Zemax、Code V）常用，按像差类型排序
    
    Attributes:
        NOLL: Noll 索引约定（默认，内部标准）
        FRINGE: Fringe（University of Arizona）索引约定
    
    References:
        - Noll, R. J. (1976). J. Opt. Soc. Am. 66(3): 207-211.
        - Wyant, J. C. & Creath, K. (1992). "Basic Wavefront Aberration Theory"
    """
    NOLL = "noll"
    FRINGE = "fringe"
    

# ============================================================================
# Fringe 索引到 Noll 索引的映射表（修正版）
# 
# 标准 Noll 序列（JOSA 1976, Noll）前 16 项：
#   Z1: Piston
#   Z2: Tilt X (ρ cos θ)
#   Z3: Tilt Y (ρ sin θ)
#   Z4: Defocus (2ρ² - 1)
#   Z5: Astigmatism 0° (ρ² cos 2θ)
#   Z6: Astigmatism 45° (ρ² sin 2θ)
#   Z7: Coma X ((3ρ³ - 2ρ) cos θ)
#   Z8: Coma Y ((3ρ³ - 2ρ) sin θ)
#   Z9: Trefoil X (ρ³ cos 3θ)
#   Z10: Trefoil Y (ρ³ sin 3θ)
#   Z11: Primary Spherical (6ρ⁴ - 6ρ² + 1)
#   ...
#
# Fringe（Zemax）序列前 16 项：
#   Z1: Piston
#   Z2: Tilt Y
#   Z3: Tilt X
#   Z4: Defocus
#   Z5: Astigmatism 45°
#   Z6: Astigmatism 0°
#   Z7: Coma Y
#   Z8: Coma X
#   Z9: Primary Spherical  <-- 注意：Fringe Z9 = Noll Z11
#   Z10: Trefoil Y
#   Z11: Trefoil X          <-- 注意：Fringe Z11 = Noll Z9

# ============================================================================
FRINGE_TO_NOLL_MAP: Dict[int, int] = {
    1: 1,    # Piston
    2: 3,    # Tilt Y: Fringe Z2 -> Noll Z3
    3: 2,    # Tilt X: Fringe Z3 -> Noll Z2
    4: 4,    # Defocus
    5: 6,    # Astigmatism 45°: Fringe Z5 -> Noll Z6
    6: 5,    # Astigmatism 0°: Fringe Z6 -> Noll Z5
    7: 8,    # Coma Y: Fringe Z7 -> Noll Z8
    8: 7,    # Coma X: Fringe Z8 -> Noll Z7
    9: 11,   # Primary Spherical: Fringe Z9 -> Noll Z11 ★ 关键修正
    10: 10,  # Trefoil Y: Fringe Z10 -> Noll Z10
    11: 9,   # Trefoil X: Fringe Z11 -> Noll Z9 ★ 关键修正
    12: 13,  # Secondary Astigmatism 45°
    13: 12,  # Secondary Astigmatism 0°
    14: 15,  # Secondary Coma Y
    15: 14,  # Secondary Coma X
    16: 22,  # Secondary Spherical: Fringe Z16 -> Noll Z22
    17: 18,  # Tetrafoil 22.5°
    18: 17,  # Tetrafoil 0°
    19: 20,  # Secondary Trefoil Y
    20: 19,  # Secondary Trefoil X
    21: 24,  # Tertiary Coma Y
    22: 23,  # Tertiary Coma X
    23: 26,  # Pentafoil Y
    24: 25,  # Pentafoil X
    25: 37,  # Tertiary Spherical: Fringe Z25 -> Noll Z37
    # 高阶项...
}



class ZernikeGenerator:
    """
    Zernike 多项式生成器（标准 Noll 索引，含 RMS 归一化）。
    
    严格按照 Noll (1976) 标准实现 Zernike 多项式，包含正确的 RMS 归一化因子，
    使得每个多项式在单位圆上的 RMS 值为 1。
    
    支持两种索引约定：
    - NOLL: 内部标准，直接使用
    - FRINGE: 自动映射到 Noll 索引
    
    Attributes:
        x (np.ndarray): 归一化光瞳 x 坐标（无量纲，范围 [-1, 1]）。
        y (np.ndarray): 归一化光瞳 y 坐标（无量纲，范围 [-1, 1]）。
        convention (ZernikeConvention): 索引约定（NOLL 或 FRINGE）。
    
    Note:
        - 像差系数单位为波数（waves），即相位延迟的波长倍数
        - 所有多项式包含 RMS 归一化因子 √(n+1) 或 √(2(n+1))
    
    References:
        - Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence". 
          J. Opt. Soc. Am. 66(3): 207-211.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 convention: ZernikeConvention = ZernikeConvention.NOLL):
        self.x = x
        self.y = y
        self.convention = convention
        self._cache: Dict[int, np.ndarray] = {}
        
        # 预计算常用量
        rho_sq = x**2 + y**2
        rho = np.sqrt(rho_sq)
        rho_4 = rho_sq**2
        rho_6 = rho_sq**3
        
        # RMS 归一化因子
        S2 = math.sqrt(2)
        S3 = math.sqrt(3)
        S5 = math.sqrt(5)
        S6 = math.sqrt(6)
        S7 = math.sqrt(7)
        S8 = math.sqrt(8)
        S10 = math.sqrt(10)
        S12 = math.sqrt(12)
        S14 = math.sqrt(14)
        
        # ============================================================================
        # 标准 Noll 索引 Zernike 多项式（含 RMS 归一化因子）
        # 公式来源：Noll, R. J. (1976). JOSA 66(3): 207-211, Table 1
        # 
        # 归一化：∫∫ |Z_n|² dA / π = 1（单位圆上 RMS = 1）
        # ============================================================================
        self._polynomial_map: Dict[int, Callable[[], np.ndarray]] = {
            # n=0 (Piston)
            1: lambda: np.ones_like(x),
            
            # n=1 (Tilt)
            2: lambda: 2 * x,                                    # Z2: Tilt X, √4·ρ·cos(θ)
            3: lambda: 2 * y,                                    # Z3: Tilt Y, √4·ρ·sin(θ)
            
            # n=2 (Defocus, Astigmatism)
            4: lambda: S3 * (2*rho_sq - 1),                      # Z4: Defocus, √3·(2ρ²-1)
            5: lambda: S6 * (x**2 - y**2),                       # Z5: Astig 0°, √6·ρ²·cos(2θ)
            6: lambda: S6 * (2*x*y),                             # Z6: Astig 45°, √6·ρ²·sin(2θ)
            
            # n=3 (Coma, Trefoil)
            7: lambda: S8 * (3*rho_sq - 2) * x,                  # Z7: Coma X, √8·(3ρ³-2ρ)·cos(θ)
            8: lambda: S8 * (3*rho_sq - 2) * y,                  # Z8: Coma Y, √8·(3ρ³-2ρ)·sin(θ)
            9: lambda: S8 * (x**3 - 3*x*y**2),                   # Z9: Trefoil X, √8·ρ³·cos(3θ)
            10: lambda: S8 * (3*x**2*y - y**3),                  # Z10: Trefoil Y, √8·ρ³·sin(3θ)
            
            # n=4 (Spherical, Secondary Astigmatism, Tetrafoil)
            11: lambda: S5 * (6*rho_4 - 6*rho_sq + 1),           # Z11: Primary Spherical ★
            12: lambda: S10 * (4*rho_sq - 3) * (x**2 - y**2),    # Z12: Sec. Astig 0°
            13: lambda: S10 * (4*rho_sq - 3) * (2*x*y),          # Z13: Sec. Astig 45°
            14: lambda: S10 * (x**4 - 6*x**2*y**2 + y**4),       # Z14: Tetrafoil 0°
            15: lambda: S10 * (4*x**3*y - 4*x*y**3),             # Z15: Tetrafoil 22.5°
            
            # n=5 (Secondary Coma, Secondary Trefoil, Pentafoil)
            16: lambda: S12 * (10*rho_4 - 12*rho_sq + 3) * x,    # Z16: Sec. Coma X
            17: lambda: S12 * (10*rho_4 - 12*rho_sq + 3) * y,    # Z17: Sec. Coma Y
            18: lambda: S12 * (5*rho_sq - 4) * (x**3 - 3*x*y**2),# Z18: Sec. Trefoil X
            19: lambda: S12 * (5*rho_sq - 4) * (3*x**2*y - y**3),# Z19: Sec. Trefoil Y
            20: lambda: S12 * (x**5 - 10*x**3*y**2 + 5*x*y**4),  # Z20: Pentafoil X
            21: lambda: S12 * (5*x**4*y - 10*x**2*y**3 + y**5),  # Z21: Pentafoil Y
            
            # n=6 (Secondary Spherical, Tertiary Astigmatism, ...)
            22: lambda: S7 * (20*rho_6 - 30*rho_4 + 12*rho_sq - 1),  # Z22: Sec. Spherical
            23: lambda: S14 * (15*rho_4 - 20*rho_sq + 6) * (x**2 - y**2),  # Z23: Tert. Astig 0°
            24: lambda: S14 * (15*rho_4 - 20*rho_sq + 6) * (2*x*y),        # Z24: Tert. Astig 45°
            
            # 高阶项（简化实现，仅提供常用项）
            37: lambda: 3 * (70*rho_6*rho_sq - 140*rho_6 + 90*rho_4 - 20*rho_sq + 1),  # Z37: Tert. Spherical
        }

    def get(self, n: int) -> np.ndarray:
        """
        获取指定索引的 Zernike 多项式。
        
        Args:
            n (int): Zernike 索引。若使用 FRINGE 约定，输入 Fringe 索引；
                     若使用 NOLL 约定（默认），输入 Noll 索引。
        
        Returns:
            np.ndarray: 对应索引的 Zernike 多项式值。
        """
        # Fringe -> Noll 映射
        noll_index = n
        if self.convention == ZernikeConvention.FRINGE:
            if n in FRINGE_TO_NOLL_MAP:
                noll_index = FRINGE_TO_NOLL_MAP[n]
            else:
                warnings.warn(
                    f"Fringe 索引 {n} 超出映射范围，将直接使用该值作为 Noll 索引",
                    UserWarning, stacklevel=2
                )
        
        if noll_index in self._cache:
            return self._cache[noll_index]
        
        calculator = self._polynomial_map.get(noll_index, lambda: np.zeros_like(self.x))
        result = calculator()
        self._cache[noll_index] = result
        return result


class OpticalSystem:
        """
        光学投影物镜系统模型（支持浸没式光刻）。
        
        物理正确性改进：
        1. 显式折射率参数，支持浸没式光刻 (NA > 1.0)
        2. 高 NA 离焦警告（建议使用 k_z 传播算子）
        3. 标准 Noll 索引 Zernike 多项式
        4. 高 NA 切趾效应 (Apodization) - 满足正弦条件的能量守恒修正
        
        Attributes:
            na (float): 物镜数值孔径 NA（无量纲）。
            wavelength_nm (float): 曝光波长（单位：nm）。
            refractive_index (float): 像空间介质折射率 n（无量纲）。
                - 空气/真空：n ≈ 1.0
                - 水浸没：n ≈ 1.44 @ 193nm
            grid_size (int): 计算网格尺寸（像素数）。
            pitch (float): 仿真区域物理尺寸（单位：nm）。
            aberrations (Dict[str, float]): 像差系数字典（单位：waves）。
            enable_apodization (bool): 是否启用高 NA 切趾修正。
        
        Example:
            >>> # 浸没式光刻 (NA=1.35, n=1.44)
            >>> optical_sys = OpticalSystem(
            ...     na=1.35,
            ...     wavelength_nm=193.0,
            ...     refractive_index=1.44,  # 水浸没
            ...     grid_size=512,
            ...     pitch=1000.0,
            ...     aberrations={'z11': 0.02},  # 球差（标准 Noll Z11）
            ...     enable_apodization=True   # 高 NA 切趾修正
            ... )
        
        Note:
            - NA 必须满足 NA ≤ n（物理约束）
            - 对于高 NA (> 0.8)，使用 Zernike Z4 离焦会引入人为球差
            - 切趾效应对 NA > 0.7 的系统对比度影响显著
        
        References:
            - Goodman, J. W. "Introduction to Fourier Optics", 4th Ed.
            - Gibson & Lanni (1992). "Experimental test of an analytical model"
            - Born & Wolf, "Principles of Optics", Chapter 8 (Abbe Sine Condition)
        """
        def __init__(self, 
                    na: float, 
                    wavelength_nm: float,
                    grid_size: int,
                    pitch: float,
                    refractive_index: float = 1.44,
                    aberrations: Optional[Dict[str, float]] = None,
                    enable_apodization: bool = False):
            """
            初始化光学系统。
            
            Args:
                na (float): 物镜数值孔径（无量纲）。
                wavelength_nm (float): 曝光波长（单位：nm）。
                grid_size (int): 计算网格尺寸（像素数）。
                pitch (float): 仿真区域物理尺寸（单位：nm）。
                refractive_index (float, optional): 像空间介质折射率。
                    默认为 1.0（空气）。浸没式光刻应设为 ~1.44。
                aberrations (Dict[str, float], optional): 像差系数字典。
                    键格式为 'zN'（Noll 索引），值为系数（单位：waves）。
                enable_apodization (bool, optional): 是否启用高 NA 切趾修正。
                    默认为 True。对于 NA > 0.7 的系统建议启用。
            
            Raises:
                ValueError: 当 NA > n 时（物理非法）。
                UserWarning: 当采样或参数设置不合理时。
            """
            # 物理合法性检查：NA 不能超过折射率
            if na > refractive_index:
                raise ValueError(
                    f"物理错误：NA ({na}) 不能超过介质折射率 n ({refractive_index})。\n"
                    f"  对于浸没式光刻，请设置 refractive_index >= {na:.2f}（如水 n≈1.44）"
                )
            
            self.na = na
            self.wavelength_nm = wavelength_nm
            self.refractive_index = refractive_index
            self.grid_size = grid_size
            self.pitch = pitch
            self.aberrations = aberrations if aberrations is not None else {}
            self.enable_apodization = enable_apodization
            # if not self.enable_apodization:
            #     print("未启用高 NA 切趾修正。")
            # 物理检查
            self._validate_sampling()
            self._validate_high_na_defocus()
            
            #
            self.specimen, self.frequency_coords, self.pupil_function = self._compute_pupil()
            debug_dir = DIAGNOSTICS_OUTPUT_DIR / "scalar_text"
            debug_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(debug_dir / "pupil_function.txt", self.pupil_function)
            # print("pupil_function.txt saved.")

        def _validate_sampling(self) -> None:
            """验证采样是否满足奈奎斯特准则。"""
            pixel_size = self.pitch / self.grid_size
            # 注意：对于浸没式，λ_eff = λ/n
            effective_wavelength = self.wavelength_nm / self.refractive_index
            nyquist_limit = effective_wavelength / (2 * (self.na / self.refractive_index))
            
            if pixel_size > nyquist_limit:
                warnings.warn(
                    f"采样不满足奈奎斯特准则！\n"
                    f"  当前像素尺寸: {pixel_size:.2f} nm\n"
                    f"  奈奎斯特极限: {nyquist_limit:.2f} nm\n"
                    f"  建议: 增大 grid_size 或减小 pitch",
                    UserWarning, stacklevel=3
                )

        def _validate_high_na_defocus(self) -> None:
            """检查高 NA 系统是否使用了 Zernike 离焦（Z4）。"""
            # 检查是否包含 z4 (Noll 离焦) 或 z4 的 Fringe 等效
            has_defocus = 'z4' in self.aberrations and self.aberrations.get('z4', 0) != 0
            
            if self.na > 0.8 and has_defocus:
                warnings.warn(
                    f"⚠️ 检测到高 NA ({self.na}) 系统使用了 Zernike 离焦 (z4)。\n"
                    f"  物理问题：对于 NA > 0.8，Zernike Z4 近似会引入人为球差。\n"
                    f"  建议方案：\n"
                    f"    1. 使用独立的 defocus_nm 参数配合 k_z 传播算子\n"
                    f"    2. 或使用精确的高 NA 离焦公式：\n"
                    f"       W = (Δz/λ) × (n - √(n² - NA²ρ²))\n"
                    f"  参考：Gibson & Lanni (1992), 'Experimental test of an analytical model'",
                    UserWarning, stacklevel=3
                )

        def _compute_pupil(self) -> tuple:
            """
            计算光瞳函数。
            
            光瞳函数包含：
            1. 孔径函数 (圆形截止)
            2. 像差引入的相位调制 (Zernike 多项式)
            3. 高 NA 切趾因子 (可选，满足 Abbe 正弦条件)
            
            切趾物理原理：
            - 满足正弦条件的物镜，平面波经高 NA 聚焦时，
            光瞳边缘的能量密度被"稀释"
            - 能量守恒要求：P_apodized(ρ) = P(ρ) / √cos(θ)
            - 其中 sin(θ) = ρ × (NA/n)
            
            Returns:
                tuple: (specimen, frequency_coords, pupil_function)
            """
            # 归一化节距计算（考虑折射率）
            # 注意：频率坐标归一化到 sin(θ) = NA/n  NA/wavelength_nm
            norm_pitch = self.pitch * self.na / self.wavelength_nm
            #根据实空间长度计算的频率坐标的采样间隔
            specimen = 1 / norm_pitch
            
            #长度依旧是grid_size
            ff_coord = np.linspace(
                -(self.grid_size-1)/2, 
                (self.grid_size-1)/2, 
                self.grid_size
            ) * specimen
            # print(f"仿真参数：grid_size: {self.grid_size}, pitch: {self.pitch}, 采样频率间隔: {specimen}\n"
            #       f"频率坐标起始值: {ff_coord[0]}, 结束值: {ff_coord[-1]}, 频率坐标长度: {ff_coord.shape}")
            
            normalized_ffx, normalized_ffy = np.meshgrid(ff_coord, ff_coord)
            
            # 频率范围验证
            max_freq = np.max(np.abs(ff_coord))
            if max_freq < 1.0:
                warnings.warn(
                    f"频率网格未覆盖完整光瞳边界！\n"
                    f"  当前最大归一化频率: {max_freq:.4f}\n"
                    f"  光瞳边界频率: 1.0\n"
                    f"  建议: 增大 pitch 或减小 grid_size",
                    UserWarning, stacklevel=3
                )
            
            # Zernike 像差计算
            zernike_gen = ZernikeGenerator(normalized_ffx, normalized_ffy)
            
            W = np.zeros_like(normalized_ffx)
            for z_index_str, coefficient in self.aberrations.items():
                try:
                    z_index = int(z_index_str.lower().replace('z', ''))
                    W += coefficient * zernike_gen.get(z_index)
                except (ValueError, TypeError):
                    warnings.warn(f"无法解析像差索引 '{z_index_str}'，已跳过。", UserWarning)

            # 光瞳孔径和相位
            r_sq = normalized_ffx**2 + normalized_ffy**2
            # 频率坐标在光瞳内的位置的值为1，否则为0
            aperture_mask = (r_sq <= 1).astype(float)
            # 光瞳内相位，单位为弧度
            phase = np.exp(1j * 2 * np.pi * W)
            
            # ====================================================================
            # 高 NA 切趾效应 (Apodization) - 满足 Abbe 正弦条件
            # 
            # 物理推导：
            #   1. 归一化频率: ρ = f / (NA/λ) = (n·sin(θ)/λ) / (NA/λ) = (n/NA)·sin(θ)
            #   2. 因此: sin(θ) = ρ × (NA/n)
            #   3. 能量守恒: 切趾因子 A(ρ) = 1 / √cos(θ)
            #
            # 数值影响 (NA=1.35, n=1.44):
            #   - ρ=1 时: sin(θ)=0.9375, θ≈69.6°, cos(θ)≈0.348
            #   - 切趾因子: 1/√0.348 ≈ 1.70 (边缘振幅增强 70%)
            # ====================================================================
            if self.enable_apodization:
                # sin(θ) = ρ × (NA/n)，其中 ρ = √r_sq
                sin_theta = np.sqrt(r_sq) * (self.na / self.refractive_index)
                
                # 数值稳定性：裁剪到 [0, 1) 避免边界处的数值问题
                # 注意：NA ≤ n 已在 __init__ 中检查，此处 sin_theta_max = NA/n < 1
                sin_theta = np.clip(sin_theta, 0, 1.0 - 1e-10)
                
                # cos(θ) = √(1 - sin²(θ))
                cos_theta = np.sqrt(1.0 - sin_theta**2)
                
                # 切趾因子: 1/√cos(θ)
                # 使用 np.maximum 而非 +1e-9 以保证精度
                apodization_factor = 1.0 / np.sqrt(np.maximum(cos_theta, 1e-10))
                
                # 仅在光瞳孔径内应用切趾
                pupil_pattern = aperture_mask * phase * apodization_factor
            else:
                # 不启用切趾时，使用均匀振幅
                pupil_pattern = aperture_mask * phase
            
            return specimen, ff_coord, pupil_pattern
