import numpy as np
import psutil
from scipy.ndimage import map_coordinates,distance_transform_edt,binary_dilation,label
from skimage.morphology import medial_axis
from matplotlib import pyplot as plt
import skimage.measure as skm
class SRAFRenderer():
    def __init__(self, SRAFs:np.ndarray, initial_width:float,samples:int,show:bool):
        self.SRAFs = SRAFs   #提供SRAF的位置
        self.initial_width = initial_width
        self.samples = samples
        self.show = False
       
    def extract_sraf_by_order_fixed(self, full_mask, target_mask, max_order=4, search_dist=20):
            """
            按阶数提取SRAF：基于连通域生长，解决跨距识别和图形破碎问题
            return: sraf_list
            """
            # 1. 提取纯SRAF区域（调用你之前定义的 overlap 判定函数）
            _, sraf_only = self.extract_sraf(full_mask, target_mask)
            
            # 类型转换
            full_mask = full_mask.astype(bool)
            target_mask = target_mask.astype(bool)
            sraf_only = sraf_only.astype(bool)
            
            # 2. 连通域标记：给每个独立的SRAF块分配唯一ID
            sraf_labels, num_features = label(sraf_only)
            
            # 初始生长前沿为主图形
            current_front = target_mask.copy()
            found_ids = set([0]) # 已识别的ID（0为背景）
            sraf_layers = {}
            all_srafs = np.zeros_like(full_mask, dtype=int)

            for order in range(1, max_order + 1):
                # 3. 膨胀搜索：向外扩张以跨越主图形与SRAF之间的间隙
                expanded = binary_dilation(current_front, iterations=search_dist)
                
                # 4. 碰撞检测：寻找膨胀区域覆盖到的SRAF标签
                hit_mask = expanded & (sraf_labels > 0) 
                if not np.any(hit_mask):
                    break
                    
                # 获取碰撞到的所有SRAF ID，并过滤掉已找过的
                hit_ids = np.unique(sraf_labels[hit_mask])
                new_ids = [i for i in hit_ids if i not in found_ids]
                
                if not new_ids:
                    continue

                # 5. 完整提取：根据ID提取完整的SRAF连通块（非破碎像素）
                current_order_mask = np.isin(sraf_labels, new_ids)
                sraf_layers[order] = current_order_mask.astype(int)
                # --- 新增功能：合并到总版图 ---
                all_srafs[current_order_mask] = 1
                # 6. 更新状态：将新发现的SRAF加入下一轮生长的前沿
                found_ids.update(new_ids)
                current_front = current_front | current_order_mask

            return sraf_layers , all_srafs, target_mask
     
     
     
    #1. MSAA 子像素 offsets 
    def generate_sample_offsets(self, samples=4):
        offsets = np.linspace(
            -(samples - 1) / (2 * samples),
            (samples - 1) / (2 * samples),
            samples
        )
        oy, ox = np.meshgrid(offsets, offsets)
        return np.stack([oy.ravel(), ox.ravel()], axis=1) 
        
    # 2. 距离阈值判定（向量化 + batch）
    # =========================================================
    def vectorized_is_within_width(self,
        dist_map,
        points,
        half_width,
        batch_size=None,
        max_mem_fraction=0.2
    ):
        points = np.asarray(points)
        N = points.shape[0]
        result = np.zeros(N, dtype=bool)

        if batch_size is None:
            total_mem = psutil.virtual_memory().total
            bytes_per_point = 16  # y,x float64
            max_bytes = total_mem * max_mem_fraction
            batch_size = max(1000, int(max_bytes // bytes_per_point))
            batch_size = min(batch_size, N)

        for i in range(0, N, batch_size):
            batch = points[i:i+batch_size]
            y = batch[:, 0]
            x = batch[:, 1]

            d = map_coordinates(
                dist_map,
                [y, x],
                order=1,
                mode='nearest'
            )

            result[i:i+batch_size] = d <= half_width

        return result
    
    #3 单个 SRAF 的距离 MSAA
    def rasterize_distance_msaa_single(self,
        sraf_mask,
        half_width,
        sample_offsets,
        batch_size=None,
        max_mem_fraction=0.2
    ):
        H, W = sraf_mask.shape

        # --- 中轴线 ---
        centerline, _ = medial_axis(sraf_mask, return_distance=True)


        # --- 到中轴线的距离 ---
        dist_map = distance_transform_edt(~centerline)

        # --- 窄带 ---
        r_pixel = np.sqrt(2) / 2
        inside = dist_map <= (half_width - r_pixel)
        outside = dist_map >= (half_width + r_pixel)
        band = ~(inside | outside)

        coverage = np.zeros((H, W), dtype=np.float32)
        coverage[inside] = 1.0

        rows, cols = np.nonzero(band)
        if len(rows) == 0:
            return coverage

        # --- 子像素坐标 ---
        S = sample_offsets.shape[0]
        samples_y = rows[:, None] + sample_offsets[:, 0]
        samples_x = cols[:, None] + sample_offsets[:, 1]
        samples = np.stack([samples_y, samples_x], axis=-1).reshape(-1, 2)

        inside_sub = self.vectorized_is_within_width(
            dist_map,
            samples,
            half_width,
            batch_size=batch_size,
            max_mem_fraction=max_mem_fraction
        )

        inside_sub = inside_sub.reshape(len(rows), S)
        coverage[rows, cols] = inside_sub.mean(axis=1)

        return coverage
    
    # 4. 多 SRAF 独立 MSAA（主入口）
    def distance_msaa_multi_sraf(
        self,
        full_sraf_mask,
        sraf_width_array,
        target,
        samples=4,
        show=False
    ):
        # ===== 强制 SRAF width 扁平化 & 标量化 =====
        sraf_width_array = np.asarray(sraf_width_array)

        if sraf_width_array.ndim > 1:
            # 比如 [[w1], [w2], [w3]] 或 [array([w1]), array([w2])]
            sraf_width_array = sraf_width_array.reshape(-1)

        sraf_width_array = [float(w) for w in sraf_width_array]

        full_sraf_mask = full_sraf_mask.astype(bool)
        H, W = full_sraf_mask.shape
        final_coverage = np.zeros((H, W), dtype=np.float32)

        sample_offsets = self.generate_sample_offsets(samples)

        # --------------------------------------------------
        # Step 1: 全 SRAF 连通域（空间基准）
        # --------------------------------------------------
        labeled, num = skm.label(full_sraf_mask, connectivity=2, return_num=True)

        # --------------------------------------------------
        # 情况 A：所有 SRAF 同一宽度
        # --------------------------------------------------
        if len(sraf_width_array) == 1:
            hw = float(sraf_width_array[0])

            for idx in range(1, num + 1):
                sraf = (labeled == idx)
                if sraf.sum() < 3:
                    continue

                coverage = self.rasterize_distance_msaa_single(
                    sraf,
                    hw,
                    sample_offsets
                )
                final_coverage = np.maximum(final_coverage, coverage)

        # --------------------------------------------------
        # 情况 B：不同阶 → 不同宽度
        # --------------------------------------------------
        else:
            # 提取分阶 SRAF（你已有函数）
            sraf_layers, _ ,target_mask1 = self.extract_sraf_by_order_fixed(
                full_mask=full_sraf_mask,
                target_mask= target,  # ⚠️ 你已有的主图形
                max_order=len(sraf_width_array)
            )

            for order in range(1, len(sraf_width_array) + 1):
                hw = float(sraf_width_array[order - 1])

                if order not in sraf_layers:
                    continue

                order_mask = sraf_layers[order].astype(bool)

                # 找到这一阶中涉及的“全局连通块”
                hit_labels = np.unique(labeled[order_mask])
                hit_labels = hit_labels[hit_labels > 0]

                for idx in hit_labels:
                    sraf = (labeled == idx)
                    if sraf.sum() < 3:
                        continue

                    coverage = self.rasterize_distance_msaa_single(
                        sraf,
                        hw,
                        sample_offsets
                    )
                    final_coverage = np.maximum(final_coverage, coverage)

        # --------------------------------------------------
        # 可视化（可选）
        # --------------------------------------------------
        if show:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            ax[0].imshow(full_sraf_mask, cmap='gray')
            ax[0].set_title('Full SRAF Mask')

            ax[1].imshow(final_coverage, cmap='gray', vmin=0, vmax=1)
            ax[1].set_title('Distance MSAA')

            ax[2].imshow(final_coverage > 0.5, cmap='gray')
            ax[2].set_title('Binary')

            for a in ax:
                a.axis('off')

            plt.tight_layout()
            plt.show()

        return final_coverage