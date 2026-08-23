# import numpy as np
# from .demo_extract_key_points import get_geometric_key_points_adaptive

# class Mask:
#     """
#     负责管理掩模版的数据、尺寸、坐标系和频谱。
#     """
#     def __init__(self, initial_mask_data: np.ndarray, pixel_size: float):
#         self._raw_data = initial_mask_data.astype(np.float64)
#         self.pixel_size = pixel_size

#         # 步骤1: 计算 pitch 的基准尺寸
#         pitch_base_size = self._get_pitch_base_size(max(self._raw_data.shape))

#         # 步骤2: 计算与原始代码完全一致的 pitch
#         # 当输入为256x256，pixel_size=4.0时，pitch = 4.0 * 256 = 1024
#         self.pitch = self.pixel_size * pitch_base_size

#         # 步骤3: 从 pitch 推导出最终的仿真网格尺寸 (grid_size)
#         self.grid_size = pitch_base_size + 1


#         # 步骤 4: 基于计算出的 pitch 和 grid_size，生成坐标系
#         self.coordinates = np.linspace(-self.pitch/2, self.pitch/2, self.grid_size)

#         # 步骤 5: 构造最终的像素化掩模 (PixelMask)
#         self.data = self._create_pixel_mask()

#         # 步骤 6: 计算频谱
#         self.spectrum = self._compute_spectrum()
         
#         # 步骤 7: 提取EPE points 
#         self._corner_points = None
#         self._midpoints = None    # EPE points

#     def _get_pitch_base_size(self, size: int) -> int:
#         """
#         私有辅助函数：精确模拟原始 find_smallest_two_expo 的行为。
#         找到小于或等于 size 的最大2的幂，或者大于size的最小2的幂，取决于原始逻辑。
#         原始逻辑 'while expo < size: expo *= 2' 对于 size=256, 返回 256。
#         """
#         expo = 1
#         while expo < size:
#             expo *= 2
#         return expo

#     def _create_pixel_mask(self) -> np.ndarray:
#         """
#         私有方法，将原始掩模数据放置在最终的仿真网格中心。
#         """
#         xsize_raw, ysize_raw = self._raw_data.shape
#         if xsize_raw % 2 == 0: xsize_raw += 1
#         if ysize_raw % 2 == 0: ysize_raw += 1
        
#         temp = np.zeros((xsize_raw, ysize_raw))
#         temp[:self._raw_data.shape[0], :self._raw_data.shape[1]] = self._raw_data
        
#         midp = (self.grid_size - 1) // 2
#         midx_raw = (xsize_raw - 1) // 2
#         midy_raw = (ysize_raw - 1) // 2
        
#         pixel_mask = np.zeros((self.grid_size, self.grid_size))
#         start_x, end_x = midp - midx_raw, midp + midx_raw + 1
#         start_y, end_y = midp - midy_raw, midp + midy_raw + 1
#         pixel_mask[start_x:end_x, start_y:end_y] = temp
        
#         return pixel_mask

#     def _compute_spectrum(self) -> np.ndarray:
#         """计算实例的频谱。"""
#         return Mask.calculate_spectrum_from_data(self.data)

#     @staticmethod
#     def calculate_spectrum_from_data(mask_data: np.ndarray) -> np.ndarray:
#         """计算中心化频谱。"""
#         if mask_data is None:
#             raise ValueError("Input mask_data cannot be None.")
#         data_for_fft = np.fft.ifftshift(mask_data)
#         spectrum_at_corner = np.fft.fft2(data_for_fft)
#         centered_spectrum = np.fft.fftshift(spectrum_at_corner)
#         return centered_spectrum
    
    
    
#     @property
#     def corner_points(self) -> list:
#         """
#         获取掩模的角点
#         在第一次被访问时会自动计算，然后缓存结果。
#         """
#         if self._corner_points is None:
#             self._calculate_and_cache_key_points()
#         return self._corner_points

#     @property
#     def midpoints(self) -> list:
#         """
#         获取掩模的线段中点
#         在第一次被访问时会自动计算，然后缓存结果。
#         """
#         if self._midpoints is None:
#             self._calculate_and_cache_key_points()
#         return self._midpoints

        
#     def _calculate_and_cache_key_points(self, tolerance_factor: float = 0.25, 
#                            include_corners: bool = True,  
#                            include_line_midpoints: bool = True) -> tuple[list, list]:
#         """
#         提取掩模的几何关键点（角点和线段中点）。
#         此方法会自动缓存第一次计算的结果。
        
#         Args:
#             tolerance_factor: 用于计算自适应公差的比例因子。
#             include_corners: 是否包含角点。
#             include_line_midpoints: 是否包含线段中点。
            
#         Returns:
#             一个元组 (corner_points, midpoints)。
#         """
#         # 检查缓存：如果参数和上次一样，且结果已存在，则直接返回
#         print("--- 计算关键点 ---")
#         corners, midpoints = get_geometric_key_points_adaptive(
#             pixel_mask = self.data,
#             tolerance_factor=tolerance_factor,
#             include_corners=include_corners,
#             include_line_midpoints=include_line_midpoints
#         )
        
#         self._corner_points = corners
#         self._midpoints = midpoints
#         return None


import numpy as np
from .demo_extract_key_points import get_geometric_key_points_adaptive

class Mask:
    """
    负责管理掩模版的数据、尺寸、坐标系和频谱。
    """
    def __init__(self, initial_mask_data: np.ndarray, pixel_size: float, 
                 default_tolerance_factor: float = 0.25):
        """
        初始化Mask对象。
        
        Args:
            initial_mask_data (np.ndarray): 原始的2D掩模版图数据。
            pixel_size (float): 每个像素代表的物理尺寸 (单位: nm)。
            default_tolerance_factor (float): 用于计算自适应公差的默认比例因子。
                                               这是一个设计上的改进，将配置作为对象的状态。
        """
        self._raw_data = initial_mask_data.astype(np.float64)
        self.pixel_size = pixel_size
        self.default_tolerance_factor = default_tolerance_factor

        # --- 核心物理定义 (源自 method2 的正确逻辑) ---
        # 步骤 1: 计算 pitch 的基准尺寸 (2的N次幂)
        pitch_base_size = self._get_pitch_base_size(max(self._raw_data.shape))

        # 步骤 2: 计算与原始代码完全一致的 pitch，确保物理定义自洽
        # 关系: pitch = pixel_size * (grid_size - 1)
        self.pitch = self.pixel_size * pitch_base_size

        # 步骤 3: 从 pitch 推导出最终的仿真网格尺寸 (奇数)
        self.grid_size = pitch_base_size + 1
        # --- 物理定义结束 ---

        # 步骤 4: 基于计算出的 pitch 和 grid_size，生成坐标系
        self.coordinates = np.linspace(-self.pitch/2, self.pitch/2, self.grid_size)

        # 步骤 5: 构造最终的像素化掩模 (PixelMask)
        self.data = self._create_pixel_mask()

        # 步骤 6: 计算频谱
        self.spectrum = self._compute_spectrum()
         
        # 步骤 7: 初始化关键点缓存
        self._corner_points: list | None = None
        self._midpoints: list | None = None

    def _get_pitch_base_size(self, size: int) -> int:
        """私有辅助函数：找到大于或等于 size 的最小2的幂。"""
        if size == 0:
            return 1
        # 使用位运算高效计算
        return 1 << (size - 1).bit_length()

    def _create_pixel_mask(self) -> np.ndarray:
        """私有方法，将原始掩模数据放置在最终的仿真网格中心。"""
        xsize_raw, ysize_raw = self._raw_data.shape
        pixel_mask = np.zeros((self.grid_size, self.grid_size))
        
        start_x = (self.grid_size - xsize_raw) // 2
        end_x = start_x + xsize_raw
        start_y = (self.grid_size - ysize_raw) // 2
        end_y = start_y + ysize_raw
        
        pixel_mask[start_x:end_x, start_y:end_y] = self._raw_data
        
        return pixel_mask

    def _compute_spectrum(self) -> np.ndarray:
        """计算实例的频谱。"""
        return Mask.calculate_spectrum_from_data(self.data)

    @staticmethod
    def calculate_spectrum_from_data(mask_data: np.ndarray) -> np.ndarray:
        """计算中心化频谱。"""
        if mask_data is None:
            raise ValueError("Input mask_data cannot be None.")
        data_for_fft = np.fft.ifftshift(mask_data)
        spectrum_at_corner = np.fft.fft2(data_for_fft)
        centered_spectrum = np.fft.fftshift(spectrum_at_corner)
        return centered_spectrum
    

    @property
    def corner_points(self) -> list:
        """
        获取掩模的角点。
        在第一次被访问时会自动计算，然后缓存结果。
        """
        if self._corner_points is None:
            self._calculate_and_cache_key_points()
        return self._corner_points

    @property
    def midpoints(self) -> list:
        """
        获取掩模的线段中点。
        在第一次被访问时会自动计算，然后缓存结果。
        """
        if self._midpoints is None:
            self._calculate_and_cache_key_points()
        return self._midpoints

    def _estimate_min_feature_size(self) -> int:
        """
        估算掩模版图的最小特征尺寸（单位：像素）。
        实现自适应公差。
        """
        mask = self._raw_data
        if mask.size == 0:
            return 0
        
        min_feature = float('inf')

        # 检查水平和垂直方向
        for axis in range(2):
            # 沿指定轴计算边缘
            edges = np.diff(mask, axis=axis, append=mask[-1:, :] if axis==0 else mask[:, -1:])
            # 找到非零边缘的位置
            edge_indices = np.where(edges != 0)
            
            for i in range(mask.shape[1-axis]):
                # 获取当前行/列的边缘位置
                line_edges = edge_indices[1-axis][edge_indices[axis] == i]  #取出第i行/列的边缘位置，再做差分可以得到边缘之间的距离
                if len(line_edges) > 1:
                    # 计算边缘之间的距离
                    distances = np.diff(line_edges)
                    min_feature = min(min_feature, np.min(distances))
        
        if min_feature == float('inf'):
            return max(mask.shape)  # 如果没有特征（纯色），返回最大尺寸
        
        return int(min_feature)

    def _calculate_and_cache_key_points(self) -> None:
        """
        提取并缓存几何关键点。
        """
        print("--- 首次计算关键点 ---")
        
        # 步骤 1: 估算最小特征尺寸 (像素单位)
        min_feature_pixels = self._estimate_min_feature_size()
        
        # 步骤 2: 计算自适应公差 (物理单位 nm)
        # 这部分计算逻辑依然很有价值，可以用于打印日志或调试，但不再直接传递给函数
        min_feature_nm = min_feature_pixels * self.pixel_size
        adaptive_tolerance = min_feature_nm * self.default_tolerance_factor
        
        print(f"最小特征尺寸: {min_feature_pixels} pixels ({min_feature_nm:.2f} nm), "
              f"使用公差因子: {self.default_tolerance_factor}")
        
        corners, midpoints = get_geometric_key_points_adaptive(
            pixel_mask=self.data,
            tolerance_factor=self.default_tolerance_factor
        )
        
        self._corner_points = corners
        self._midpoints = midpoints
