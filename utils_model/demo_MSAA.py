import numpy as np
import psutil
class   AntiAliasRenderer:
        def __init__(self, msaa_level=4):
            """
            抗锯齿渲染器初始化

            参数：
            msaa_level (int): 多重采样抗锯齿等级，支持4或16
            """
            self.sample_offsets = self._get_sample_offsets(msaa_level)
        
        def _generate_6x6_offsets(self):
            """生成旋转的6x6采样点模式"""
        # 基础网格参数
            grid_size = 6
            rotation_angle = np.deg2rad(15)  # 旋转15度避免规则图案
        
        # 生成基础网格坐标 (范围：0~1)
            x = np.linspace(0.0833, 0.9167, grid_size)  # 6等分避开边缘
            y = np.linspace(0.0833, 0.9167, grid_size)
            xx, yy = np.meshgrid(x, y)
        
        # 转换为偏移量（中心坐标系）
            offsets = np.stack([xx.ravel() - 0.5, 
                           yy.ravel() - 0.5], axis=1)
        
        # 应用旋转矩阵
            rot_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
            ])
            rotated_offsets = np.dot(offsets, rot_matrix.T)
        
        # 限制偏移范围到±0.4375像素
            return (rotated_offsets * 0.875).astype(np.float32)  # 0.5 * 0.875 = 0.4375
        
        def _generate_8x8_offsets(self):
            """生成8x8 MSAA采样点偏移，格式为[y, x]"""
            # 基础参数
            grid_size = 8
            rotation_angle = np.deg2rad(15)  # 旋转15度破坏规则性
            base_scale = 0.4375  # 控制偏移范围在±0.4375像素内
    
            # 生成均匀网格（y轴优先）
            y = np.linspace(0.5/(grid_size+1), 1 - 0.5/(grid_size+1), grid_size)
            x = np.linspace(0.5/(grid_size+1), 1 - 0.5/(grid_size+1), grid_size)
            yy, xx = np.meshgrid(y, x, indexing='ij')  # 'ij'索引确保y轴优先
    
            # 转换为中心坐标系（-0.5~0.5）
            offsets = np.stack([yy.ravel() - 0.5, 
                        xx.ravel() - 0.5], axis=1)  # [y, x]
    
            # 应用旋转矩阵（注意轴顺序）
            rot_matrix = np.array([
                [np.cos(rotation_angle), np.sin(rotation_angle)],  # y轴旋转分量
                [-np.sin(rotation_angle), np.cos(rotation_angle)]   # x轴旋转分量
                ])
            rotated_offsets = np.dot(offsets, rot_matrix.T)
    
            # 限制偏移范围并归一化
            scaled_offsets = rotated_offsets * base_scale * 2  # 缩放至±0.4375
            # plt.figure(figsize=(6,6))
            # plt.scatter(scaled_offsets[:,1], scaled_offsets[:,0], s=10, c='red')  # 注意x和y轴的交换
            # plt.title('8x8 MSAA采样点分布 ([y, x]格式)')
            # plt.xlabel('x偏移')
            # plt.ylabel('y偏移')
            # plt.xlim(-0.5, 0.5)
            # plt.ylim(-0.5, 0.5)
            # plt.grid(True)
            # plt.show()
            
            
            return scaled_offsets.astype(np.float32)
        def generate_8x8_uniform_offsets_yx(self):
            """生成8x8均匀采样点偏移，格式为[y, x]，无旋转"""
            # 基础参数
            grid_size = 8
            base_scale = 0.4375  # 偏移范围±0.4375像素
    
            # 生成均匀网格（y轴优先）
            y = np.linspace(0.5/(grid_size+1), 1 - 0.5/(grid_size+1), grid_size)
            x = np.linspace(0.5/(grid_size+1), 1 - 0.5/(grid_size+1), grid_size)
            yy, xx = np.meshgrid(y, x, indexing='ij')  # 'ij'索引确保y轴优先
    
            # 转换为中心坐标系（-0.5~0.5）并缩放
            offsets_y = (yy.ravel() - 0.5) * base_scale * 2  # y方向缩放
            offsets_x = (xx.ravel() - 0.5) * base_scale * 2  # x方向缩放
    
            # 组合为[y, x]格式
            return np.column_stack((offsets_y, offsets_x)).astype(np.float32)
        
        def _get_sample_offsets(self, level):
            if level == 4:
            # 4x MSAA采样模式（旋转式分布）
                return np.array([
                [-0.25, -0.25],  # 左上
                [-0.25, 0.25],   # 左下
                [0.25, 0.25],    # 右下 
                [0.25, -0.25]    # 右上
            ], dtype=np.float32)
            elif level == 16:
            # 16x MSAA采样模式（4x4均匀网格）
                return np.array([
                # 第一行(y-0.375)
                [-0.375, -0.375], [-0.125, -0.375], [0.125, -0.375], [0.375, -0.375],
                # 第二行(y-0.125)
                [-0.375, -0.125], [-0.125, -0.125], [0.125, -0.125], [0.375, -0.125],
                # 第三行(y+0.125)
                [-0.375, 0.125],  [-0.125, 0.125],  [0.125, 0.125],  [0.375, 0.125],
                # 第四行(y+0.375)
                [-0.375, 0.375],  [-0.125, 0.375],  [0.125, 0.375],  [0.375, 0.375]
            ], dtype=np.float32)
            
            elif level == 36:
                return self._generate_6x6_offsets()
            elif level == 64:
                return self.generate_8x8_uniform_offsets_yx()
            
            else:
                raise ValueError("仅支持4x或16x MSAA")
        
        def vectorized_is_inside(self, polygon, points, batch_size=None, max_mem_fraction=0.2):
            """
            判断采样点是否在多边形内 (Ray Casting)，支持自动分批
            polygon 曲线上的点
            points 网格细分的采样点
            """
            polygon = np.asarray(polygon)
            points = np.asarray(points)
            if polygon.shape[0] < 3:
                return np.zeros(points.shape[0], dtype=bool)

            n = polygon.shape[0]
            p1 = polygon
            p2 = np.roll(polygon, -1, axis=0)

            # === 自动 batch_size 计算 ===
            if batch_size is None:
                total_mem = psutil.virtual_memory().total
                bytes_per_point = n * 8  # 每个点 × 边数 × float64
                max_bytes = total_mem * max_mem_fraction
                batch_size = max(1000, int(max_bytes // bytes_per_point))
                batch_size = min(batch_size, len(points))

            result = np.zeros(points.shape[0], dtype=bool)

            # === 分批处理 ===
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                y, x = batch[:, 0], batch[:, 1]

                with np.errstate(divide='ignore', invalid='ignore'):
                    y_gt = (p1[:, 0] > y[:, None]) != (p2[:, 0] > y[:, None])
                    t = (y[:, None] - p1[:, 0]) / (p2[:, 0] - p1[:, 0] + 1e-8)
                    x_intersect = p1[:, 1] + t * (p2[:, 1] - p1[:, 1])
                    valid = (t >= 0) & (t <= 1) & y_gt
                    right_intersect = (x_intersect > x[:, None]) & valid

                intersections = np.count_nonzero(right_intersect, axis=1)
                result[i:i+batch_size] = (intersections % 2) == 1

            return result


        def rasterize_polygon(self, polygon, height, width, type='gray', batch_size=None, max_mem_fraction=0.2):
            polygon = np.asarray(polygon)
            if len(polygon) < 3:
                return np.zeros((height, width, self.sample_offsets.shape[0]), dtype=bool)

            # === 包围盒 ===
            ymin, xmin = np.floor(polygon.min(axis=0)).astype(int)
            ymax, xmax = np.ceil(polygon.max(axis=0)).astype(int)
            ymin, xmin = max(0, ymin), max(0, xmin)
            ymax, xmax = min(height, ymax + 1), min(width, xmax + 1)

            # === 网格采样 ===
            rows = np.arange(ymin, ymax)
            cols = np.arange(xmin, xmax)
            grid_y, grid_x = np.meshgrid(rows, cols, indexing='ij')

            samples_y = grid_y[:, :, None] + self.sample_offsets[:, 0]
            samples_x = grid_x[:, :, None] + self.sample_offsets[:, 1]
            samples = np.stack([samples_y, samples_x], axis=-1).reshape(-1, 2)

            # === 分批 inside 计算 ===
            inside = self.vectorized_is_inside(polygon, samples, batch_size=batch_size, max_mem_fraction=max_mem_fraction)

            # === 覆盖率 ===
            sample_coverage = inside.reshape(len(rows), len(cols), -1)

            if type == 'binary':
                full_sample_coverage = np.zeros((height, width, self.sample_offsets.shape[0]), dtype=bool)
                full_sample_coverage[ymin:ymax, xmin:xmax, :] = sample_coverage
                return full_sample_coverage
            else:
                coverage = sample_coverage.mean(axis=2)
                full_coverage = np.zeros((height, width), dtype=np.float32)
                full_coverage[ymin:ymax, xmin:xmax] = coverage
                return full_coverage


        def MSAA(self, all_polygons, mask, type='gray', batch_size=None, max_mem_fraction=0.2):
            height, width = mask.shape
            S = self.sample_offsets.shape[0]

            if type == 'binary':
                # 三维 bool 掩膜
                final_samples = np.zeros((height, width, S), dtype=bool)

                for polygon in all_polygons:
                    sample_mask = self.rasterize_polygon(polygon, height, width, type='binary',
                                                        batch_size=batch_size, max_mem_fraction=max_mem_fraction)
                    sample_mask = sample_mask.astype(bool)
                    final_samples ^= sample_mask  # 奇偶填充

                coverage = final_samples.mean(axis=2) >= 0.5
                return coverage.astype(np.float32)

            elif type == 'gray':
                final_coverage = np.zeros((height, width), dtype=np.float32)

                for polygon in all_polygons:
                    coverage = self.rasterize_polygon(polygon, height, width, type='gray',
                                                    batch_size=batch_size, max_mem_fraction=max_mem_fraction)
                    np.maximum(final_coverage, coverage, out=final_coverage)

                return final_coverage

            else:
                raise ValueError("type must be 'binary' or 'gray'")