import numpy as np
import cv2
from matplotlib import pyplot as plt



#_select_eps_ofothers：用在不是通孔的版图上选点的，有两个参数可以调
#_select_eps_ofvia：用在通孔的版图上选点，只有一个参数可以调
#效果就是文件里面的图的效果，红色的是权重高的，蓝色的是权重低的，只打ep点的话代码可以输出ep点坐标（包括红蓝所有）
class DemoSelectEps:
    def __init__(self,
                 target_mask,
                 mid_weight,
                 orther_weight,
                 pattern_name):
        """ 该类用于选择版图上的ep点
            并且可以根据输入的权重，将ep点分为主要区域和次要区域，输出一个权重矩阵，用于meef矩阵的加权计算
            或者是单独输出ep点坐标

        Args:
            target_mask (_np_): _目标版图_
            mid_weight (_type_): _主要区域的权重_
            orther_weight (_type_): _其他次要区域的权重_
            pattern_name (_type_): _版图名称_
        """
        self.target = target_mask
        self.mid_weight = mid_weight
        self.orther_weight = orther_weight
        self.pattern_name = pattern_name
        # if pattern_name == '中心对称通孔' or pattern_name == '对角通孔':
        #     self.eps,self.wepe_caculated, self.weight_mee = self._select_eps_ofvia(r =2)
        # else:
        #     self.eps,self.wepe_caculated, self.weight_mee = self._select_eps_ofothers(interval_line = 6, interval_corner=0)
         
    def fix_cps(self, ls_mask, k, symmetry="left-right"):
        """
        提取等距采样点并生成对称点集，用于直接在目标版图上选取控制点
        参数:
            ls_mask: 二值mask (numpy array)
            k: 采样点数
            symmetry: 对称类型
                "center"       - 中心对称
                "left-right"   - 左右对称
                "diagonal"     - 左下到右上的对角线对称
        """
        mask_uint8 = (ls_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_simpling_mcps = []
        
        for contour in contours:
            contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            sampling_pts = self.sample_elements(contour_pts, k, 'skip')
            all_simpling_mcps.append(sampling_pts)
        
        # 图像中心 (y, x)
        center = [ls_mask.shape[0] // 2, ls_mask.shape[1] // 2]

        if len(all_simpling_mcps) > 1:
            if symmetry == "center":
                # 中心对称
                all_simpling_mcps[1] = [
                    [2 * center[0] - y, 2 * center[1] - x] for y, x in all_simpling_mcps[0]
                ]
            elif symmetry == "left-right":
                # 左右对称（x坐标关于中心对称，y不变）
                all_simpling_mcps[1] = [
                    [y, 2 * center[1] - x] for y, x in all_simpling_mcps[0]
                ]
            elif symmetry == "diagonal":
                # 对角线对称（左下到右上）
                # 对角线方程 y + x = H-1 （H 图像高度-1）
                H = ls_mask.shape[0]
                W = ls_mask.shape[1]
                all_simpling_mcps[1] = [
                    [H - 1 - x, W - 1 - y] for y, x in all_simpling_mcps[0]
                ]
            else:
                raise ValueError("symmetry 参数必须是 'center', 'left-right' 或 'diagonal'")
        
        return all_simpling_mcps     
         
    # def fix_OPC_cps(self,ls_mask,k):
    #     mask_uint8 = (ls_mask * 255).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     all_mcps = []
    #     all_simpling_mcps = []
        
    #     for contour in contours:
    #         contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            
    #         sampling_pts = self.sample_elements(contour_pts, k, 'skip')
            
    #         all_simpling_mcps.append(sampling_pts)
        
       
            
                
    #     return all_simpling_mcps     
           
    def _select_eps_ofothers(self,interval_line:int = 6, interval_corner:int =0):
        """
        提取 Mask 轮廓上的采样点作为EP点，并为线段中间区域分配高权重。
        
        Args:
            mask: 输入的二值掩模图像
            interval_line: 线段上的采样间隔
            interval_corner: 距离角点的保留距离（不进行高权重采样的缓冲区）
            weight_val: 核心区域的权重值 (通常 > 1)
            
        Returns:
            initial_eps: 所有采样点的坐标数组 (N, 2)
            weight_meef: MEEF 权重矩阵 (1, N)，核心区域为 weight_val，其余为 1，这部分是输出一个权重矩阵，用于对MEEF矩阵加权
            weight_epe: EPE 权重矩阵 (1, N)，核心区域为 1，其余为 0，这部分是筛选出核心区域的ep点用于计算EPE
        """
        
        # 1. 预处理与轮廓提取
        mask_uint8 = (self.target * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_evaluation_points = []  # 存储所有采样点
        core_region_points = []     # 存储需要加权的核心区域点

        for contour in contours:
            # cv2 轮廓点默认为 (x, y)，这里转换为 (row, col) 格式以匹配矩阵索引
            # shape: (N, 1, 2) -> (N, 2)
            pts = contour[:, 0, :]
            pts = np.column_stack((pts[:, 1], pts[:, 0])) 
            
            num_pts = len(pts)
            for i in range(num_pts):
                start_pt = pts[i]
                end_pt = pts[(i + 1) % num_pts]
                
                # 获取线段上的所有像素坐标
                line_coords = self._bresenham_line_with_startandend(*start_pt, *end_pt)
                seg_len = len(line_coords)
                
                # 忽略过短的线段
                if seg_len < 3:
                    continue

                # === 步骤 A: 确定线段上的采样索引 ===
                # 逻辑：从中心向两边辐射采样，同时强制包含两端的“角点保护位”
                mid_idx = seg_len // 2
                left_limit = interval_corner
                right_limit = seg_len - 1 - interval_corner
                
                # 使用集合自动去重，避免中心点或端点重复
                indices_set = {mid_idx}
                
                # 向左采样
                for idx in range(mid_idx - interval_line, left_limit - 1, -interval_line):
                    indices_set.add(idx)
                # 向右采样
                for idx in range(mid_idx + interval_line, right_limit + 1, interval_line):
                    indices_set.add(idx)
                    
                # 强制添加两端保护点（如果在线段范围内）
                if left_limit < seg_len: indices_set.add(left_limit)
                if right_limit >= 0: indices_set.add(right_limit)
                
                # 排序并转为列表，同时过滤掉越界的索引
                sampled_indices = sorted([idx for idx in indices_set if 0 <= idx < seg_len])
                
                if not sampled_indices:
                    continue

                # 根据索引提取实际坐标
                current_segment_samples = [line_coords[idx] for idx in sampled_indices]
                all_evaluation_points.extend(current_segment_samples)

                # === 步骤 B: 筛选核心区域 (High Weight Region) ===
                # 逻辑：只取中间约 70% 的点作为核心加权区
                total_count = len(sampled_indices)
                target_count = int(round((total_count - 2) * 0.85))
                
                # 保证取样数量为奇数且至少为3
                if target_count % 2 == 0: 
                    target_count += 1
                target_count = max(3, target_count)
                
                # 找到中心点在采样列表中的位置
                try:
                    mid_pos_in_list = sampled_indices.index(mid_idx)
                except ValueError:
                    # 极少情况：如果mid_idx没被采样到，找最近的一个
                    mid_pos_in_list = min(range(len(sampled_indices)), key=lambda k: abs(sampled_indices[k] - mid_idx))
                
                # 计算保留半径并切片
                radius = (target_count - 1) // 2
                start_k = max(0, mid_pos_in_list - radius)
                end_k = min(len(sampled_indices), mid_pos_in_list + radius + 1)
                
                final_indices = sampled_indices[start_k : end_k]
                core_region_points.extend([line_coords[idx] for idx in final_indices])

        # 3. 结果整合与权重分配
        initial_eps = np.array(all_evaluation_points)
        if len(initial_eps) == 0:
            return np.empty((0, 2)), np.empty((1, 0)), np.empty((1, 0))

        # 使用 set 进行 O(1) 复杂度的查找 (比原代码的 np.all 快很多)
        core_points_set = set(map(tuple, core_region_points))
        
        # 初始化权重 (默认全是 1)
        num_eps = len(initial_eps)
        weight_meef = np.ones(num_eps)
        
        # 标记核心区域
        is_core = np.array([tuple(pt) in core_points_set for pt in initial_eps])
        weight_meef[is_core] = self.mid_weight
        
        # 调整形状为 (1, N)
        weight_meef = weight_meef.reshape(1, -1)
        
        # 4. 生成 EPE 权重
        # 逻辑：核心加权区设为 1，其余背景设为 0
        weight_epe = np.zeros_like(weight_meef)
        weight_epe[weight_meef == self.mid_weight] = 1.0
        
        return initial_eps,  weight_epe, weight_meef
    
    def extract_mask_control_points(self, k:int, symmetry=None):
        """
        统一提取控制点（MCP）的函数，支持普通采样和多种对称性约束采样。

        Args:
            mask: 二值化的掩模 (numpy array)
            k: 每个轮廓的采样点数
            symmetry: 对称类型，可选：
                    None           - 普通 OPC 采样 (不应用对称)
                    "center"       - 中心对称
                    "left-right"   - 左右对称
                    "diagonal"     - 对角线对称 (左下到右上)
        
        Returns:
            all_sampling_mcps: 包含所有轮廓采样点坐标的列表
        """
        # 1. 基础轮廓提取与采样
        mask_uint8 = (self.target * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        all_sampling_mcps = []
        for contour in contours:
            # 将 OpenCV 的 (x, y) 转换为坐标系 (y, x)
            contour_pts = contour[:, 0, [1, 0]].astype(np.float32)
            sampling_pts = self.sample_elements(contour_pts, k, 'skip')
            all_sampling_mcps.append(sampling_pts)

        # 2. 如果不需要对称性或者只有一个图形，直接返回
        if symmetry is None or len(all_sampling_mcps) < 2:
            return all_sampling_mcps

        # 3. 应用对称逻辑 (基于第一个图形的采样点强制约束第二个图形)
        h, w = self.target.shape
        center_y, center_x = h // 2, w // 2
        
        # 获取参考图形的点集
        ref_pts = all_sampling_mcps[0]
        
        if symmetry == "center":
            # 中心对称: (y', x') = (2*cy - y, 2*cx - x)
            new_pts = [[2 * center_y - y, 2 * center_x - x] for y, x in ref_pts]
        
        elif symmetry == "left-right":
            # 左右对称: y不变, (x') = (2*cx - x)
            new_pts = [[y, 2 * center_x - x] for y, x in ref_pts]
        
        elif symmetry == "diagonal":
            # 对角线对称 (关于 y + x = H-1 对称): (y', x') = (H-1-x, W-1-y)
            new_pts = [[h - 1 - x, w - 1 - y] for y, x in ref_pts]
        
        else:
            raise ValueError(f"不支持的对称类型: {symmetry}。请选择 'center', 'left-right', 'diagonal' 或 None。")

        # 更新第二个图形的采样点
        all_sampling_mcps[1] = new_pts

        return all_sampling_mcps
    
    def sample_elements(self,lst, k, mode="step"):
            """
            从列表中按间隔k采样元素。
            :param lst: 原始列表
            :param k: 间隔值
            :param mode: "step"（步长k）或 "skip"（步长k+1）
            :return: 采样后的列表
            """
            if mode == "step":
                return np.array(lst[::k] )      # 每k个元素取一个
            elif mode == "skip":
                return np.array(lst[::k+1])    # 每隔k个元素取一个
            else:
                raise ValueError("模式需为 'step' 或 'skip'")
    
    def _bresenham_line_with_startandend(self,y1, x1, y2, x2):
        """
        用于计算两个坐标之间的直线上的所有点
       
        """
        
        points = [(y1, x1)]  # 初始化时包含起点
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        sy = 1 if y2 > y1 else -1
        sx = 1 if x2 > x1 else -1
        err = dx - dy

        while (y1, x1) != (y2, x2):
            e2 = 2 * err
        # 调整误差和坐标
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        # 直接添加新坐标（无需判断终点）
            points.append([y1, x1])
    

        return points 
    
    def remove_adjacent_close_points(self,nested_list, threshold=2):
        """
        删除每个轮廓中相邻距离小于 threshold 的点
        :param nested_list: 形如 [[[y, x], [y, x], ...], [...], ...]
        :param threshold: 距离小于该值则认为是“挨着的”
        :return: 去重后的新列表
        """
        cleaned_contours = []

        for contour in nested_list:
            n = len(contour)
            if n == 0:
                cleaned_contours.append([])
                continue
            if n == 1:
                cleaned_contours.append(contour[:])
                continue

            result = []
            for i in range(n):
                prev_idx = (i - 1) % n  # 环绕前一个点
                dist = np.linalg.norm(np.array(contour[i]) - np.array(contour[prev_idx]))
                if dist >= threshold:
                    result.append(contour[i])
            cleaned_contours.append(result)

        return cleaned_contours
    
    def _select_eps_ofvia(self, r :int =2):
            """
            为通孔类型的版图选择EP点，选择每边中点作为ep点，返回值与上面的select_eps函数相同
            """
            mask_uint8 = (self.target * 255).astype(np.uint8)  
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cps = []   # 存储所有轮廓的边界点
            wepe_caculated = [] # 用来计算wepe的权重
            weight_meef = []    #用来赋予损失函数的权重
            for contour in contours:
                contour_pts = [[pt[0][1], pt[0][0]] for pt in contour]  # 转为 (y, x)
                length = len(contour_pts)
                sample_coords = []
                for i in range(length):
                    start = contour_pts[i]
                    end = contour_pts[(i + 1) % length]
                    coords = self._bresenham_line_with_startandend(*start, *end)

                    if len(coords) < 4:
                        continue  # 太短的不采样
                    # 添加圆角点：仅在四角的场景中适用
                    
                    if i < 4:  # 如果你是矩形
                        corner_offset = [[r, r], [-r, r], [-r, -r], [r, -r]][i]
                        cy, cx = coords[0][0] + corner_offset[0], coords[0][1] + corner_offset[1]
                        sample_coords.append([cy, cx])
                        wepe_caculated.extend([0])
                        weight_meef.extend([self.orther_weight])
                    # 添加边缘采样点（四分之一、中点、三分之一）
                    points_to_add = [
                    coords[len(coords) // 4],
                    coords[len(coords) // 3],
                    coords[len(coords) // 2],
                    coords[-1 - len(coords) // 3],
                    coords[-1 - len(coords) // 4]]
                    #构建权重列表
                    wepe_caculated.extend([0])
                    wepe_caculated.extend([0])
                    weight_meef.extend([self.orther_weight])
                    weight_meef.extend([self.orther_weight])
                    wepe_caculated.extend([1])
                    weight_meef.extend([self.mid_weight])
                    wepe_caculated.extend([0])
                    weight_meef.extend([self.orther_weight])
                    wepe_caculated.extend([0])
                    weight_meef.extend([self.orther_weight])          
                    for pt in points_to_add:
                                py, px = pt
                                sample_coords.append([py, px])
                cps.append(sample_coords)
                initial_eps = [pt for contour in cps for pt in contour]   
                initial_eps = np.asarray(initial_eps).reshape(-1,2)
            return initial_eps, wepe_caculated, weight_meef 