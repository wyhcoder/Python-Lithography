# import matlab.engine
# import os
# import pickle
# import hashlib
# import numpy as np

# from enum import Enum
# from scipy.special import erf


# # matlab驱动
# def matlab_imresize(array: np.ndarray, scale: float) -> np.ndarray:
#     """使用MATLAB的imresize函数进行图像缩放，带缓存功能
    
#     Args:
#         array: 输入数组
#         scale: 缩放比例
        
#     Returns:
#         缩放后的数组
#     """
#     # 缓存文件路径
#     cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)
#     cache_path = os.path.join(cache_dir, 'imresize_cache.pkl')
    
#     # 计算输入的哈希值
#     input_hash = hashlib.md5(array.tobytes()).hexdigest()
#     cache_key = f"{input_hash}_{scale}"
    
#     # 尝试从缓存加载
#     if os.path.exists(cache_path):
#         try:
#             with open(cache_path, 'rb') as f:
#                 cache = pickle.load(f)
#                 if cache_key in cache:
#                     return cache[cache_key]
#         except:
#             cache = {}
#     else:
#         cache = {}
    
#     # 如果没有缓存，使用MATLAB计算
#     eng = matlab.engine.start_matlab()
#     matlab_array = matlab.double(array.tolist())
#     result = np.array(eng.imresize(matlab_array, scale, 'bilinear'))
#     eng.quit()
    
#     # 保存到缓存
#     cache[cache_key] = result
#     with open(cache_path, 'wb') as f:
#         pickle.dump(cache, f)
    
#     return result 



# class SourceType(Enum):
#     """光源类型枚举"""
#     CONVENTIONAL = "conventional"
#     ANNULAR = "annular"
#     DIPOLE = "dipole"
#     QUASAR = "quasar"

# def edeta(deta: float, x: np.ndarray) -> np.ndarray:
#     """计算平滑函数"""
#     deta = max(deta, 1e-9)
#     return 0.5 * (1 + erf(x/deta))

# def _generate_source_pattern(x: np.ndarray, y: np.ndarray, source_type: str, 
#                            deta: float, sin: float, sout: float) -> np.ndarray:
#     """根据类型生成基础的光源形状"""
#     theta = np.arctan2(y, x)
#     r = np.sqrt(x**2 + y**2)
#     phi = np.pi/8
    
#     base_annulus = edeta(deta, sout - r) * edeta(deta, r - sin)
    
#     if source_type == SourceType.CONVENTIONAL.value:
#         return edeta(deta, sout - r)
#     elif source_type == SourceType.ANNULAR.value:
#         return base_annulus
#     elif source_type == SourceType.DIPOLE.value:
#         angle_term = (edeta(deta, phi - np.abs(theta)) + 
#                       edeta(deta, phi - np.abs(np.pi - np.abs(theta))))
#         return base_annulus * angle_term
#     elif source_type == SourceType.QUASAR.value:
#         angle_term = (edeta(deta, phi - np.abs(0.25*np.pi - theta)) + 
#                       edeta(deta, phi - np.abs(0.75*np.pi - theta)) +
#                       edeta(deta, phi - np.abs(-0.75*np.pi - theta)) + 
#                       edeta(deta, phi - np.abs(-0.25*np.pi - theta)))
#         return base_annulus * angle_term
    
#     return np.zeros_like(x)


# class Illumination:
#     """
#     负责根据参数生成光源强度分布图。
#     并调用MATLAB进行降采样。
#     """
#     def __init__(self,
#                  source_type: str,
#                  sigma_in: float,
#                  sigma_out: float,
#                  frequency_coords: np.ndarray,
#                  deta: float = 0.0,
#                  scale: int = 10):
#         """
#         初始化光源系统。

#         Args:
#             source_type: 光源类型
#             sigma_in: 内 coherence factor (σ_in)
#             sigma_out: 外 coherence factor (σ_out)
#             frequency_coords: 光学系统的完整频域坐标 (来自 OpticalSystem)
#             deta: 光源边缘的平滑因子
#             scale: 超采样率 (原始代码中硬编码为10)
#         """
#         self.type = source_type.lower()
#         self.sigma_in = sigma_in
#         self.sigma_out = sigma_out
#         self.frequency_coords = frequency_coords
#         self.deta = deta
#         self.scale = scale

#         self.source_map: np.ndarray
#         self.source_coords: np.ndarray
        
#         self.source_map, self.source_coords = self._compute_source_map()

#     def _compute_source_map(self) -> tuple[np.ndarray, np.ndarray]:
#         """
#         严格按照原始 compute_source 函数的逻辑计算光源图。
#         """
#         ff = self.frequency_coords
#         ff_source_ind = (ff >= -2) & (ff <= 2)
#         ff_source = ff[ff_source_ind]
        
#         num = len(ff_source)
#         temp_ffsource = np.linspace(ff_source[0], ff_source[-1], self.scale * num)
        
#         ffx, ffy = np.meshgrid(temp_ffsource, temp_ffsource)
        
#         temp_source = _generate_source_pattern(ffx, ffy, 
#                                                self.type,
#                                                self.deta,
#                                                self.sigma_in,
#                                                self.sigma_out)
        
#         # --- 关键修改 ---
#         # 调用原始的 matlab_imresize 函数进行降采样。
#         # 这将启动MATLAB引擎来执行计算。
#         source = matlab_imresize(temp_source, 1/self.scale)
#         # --------------------

#         return source, ff_source


import numpy as np
from enum import Enum
from scipy.special import erf
from PIL import Image 
from typing import Tuple, List
from utils_model.project_paths import DIAGNOSTICS_OUTPUT_DIR
class SourceType(Enum):
    """光源类型枚举"""
    CONVENTIONAL = "conventional"
    ANNULAR = "annular"
    DIPOLE = "dipole"
    QUASAR = "quasar"

def _edeta(deta: float, x: np.ndarray) -> np.ndarray:
    """计算平滑边缘函数"""
    deta = max(deta, 1e-9)
    return 0.5 * (1 + erf(x / deta))

def _generate_source_pattern(x: np.ndarray, y: np.ndarray, source_type: str, 
                           deta: float, sin: float, sout: float) -> np.ndarray:
    """根据类型在高分辨率网格上生成基础的光源形状"""
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phi = np.pi / 8
    
    base_annulus = _edeta(deta, sout - r) * _edeta(deta, r - sin)
    
    if source_type == SourceType.CONVENTIONAL.value:
        # 常规光源的 sigma_in 为 0
        return _edeta(deta, sout - r)
    elif source_type == SourceType.ANNULAR.value:
        return base_annulus
    elif source_type == SourceType.DIPOLE.value:
        angle_term = (_edeta(deta, phi - np.abs(theta)) + 
                      _edeta(deta, phi - np.abs(np.pi - np.abs(theta))))
        return base_annulus * angle_term
    elif source_type == SourceType.QUASAR.value:
        angle_term = (_edeta(deta, phi - np.abs(0.25*np.pi - theta)) + 
                      _edeta(deta, phi - np.abs(0.75*np.pi - theta)) +
                      _edeta(deta, phi - np.abs(-0.75*np.pi - theta)) + 
                      _edeta(deta, phi - np.abs(-0.25*np.pi - theta)))
        return base_annulus * angle_term
    
    return np.zeros_like(x)

class Illumination:
    """
    根据参数生成光源强度分布图。
    """
    def __init__(self,
                 source_type: str,
                 sigma_in: float,
                 sigma_out: float,
                 frequency_coords: np.ndarray,
                 deta: float = 0.0,
                 scale: int = 10):
        """
        初始化光源系统。

        Args:
            source_type: 光源类型
            sigma_in: 内 coherence factor (σ_in)
            sigma_out: 外 coherence factor (σ_out)
            frequency_coords: 光学系统的完整频域坐标 (来自 OpticalSystem)
            deta: 光源边缘的平滑因子
            scale: 超采样率 (原始代码中为10)
        """
        self.type = source_type.lower()
        self.sigma_in = sigma_in
        self.sigma_out = sigma_out
        self.frequency_coords = frequency_coords #一维的频率坐标轴
        self.deta = deta
        self.scale = scale

        self.source_map: np.ndarray
        self.source_coords: np.ndarray
        
        # 在初始化时直接计算
        self.source_map, self.source_coords = self._compute_source_map()
        debug_dir = DIAGNOSTICS_OUTPUT_DIR / "scalar_text"
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(debug_dir / "source_map.txt", self.source_map, fmt="%.4f")
        # print("source_map",self.source_map)
        #self.source_coords 频率坐标范围一维的
        #self.source_map 光源强度分布图
    def _compute_source_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算光源图。
        """
        # 1. 筛选频域坐标
        ff = self.frequency_coords
        ff_source_ind = (ff >= -2) & (ff <= 2)
        ff_source = ff[ff_source_ind]
        
        # 2. 创建高分辨率坐标轴
        num = len(ff_source)
        temp_ffsource = np.linspace(ff_source[0], ff_source[-1], self.scale * num)
        
        # 3. 生成高分辨率网格
        ffx, ffy = np.meshgrid(temp_ffsource, temp_ffsource)
        
        # 4. 计算高分辨率光源分布
        temp_source = _generate_source_pattern(ffx, ffy, 
                                               self.type,
                                               self.deta,
                                               self.sigma_in,
                                               self.sigma_out)
        
        final_size = (int(temp_source.shape[1] / self.scale), int(temp_source.shape[0] / self.scale))
        source = np.array(Image.fromarray(temp_source).resize(final_size, Image.Resampling.BILINEAR))
        
        return source, ff_source  
