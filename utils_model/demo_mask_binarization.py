import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk
from typing import Tuple, List


def binarize_mask(
    gray_mask: np.ndarray, 
    target_pattern: np.ndarray,
    quantile_val: float = 0.125,
    area_threshold: int = 100,
    peak_min_intensity: float = 0.1,
    min_spacing_pixels: int = 5 # SRAF与主图形的最小间距（像素数），必须有数，不然结果没有意义！！！！
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用一个多阶段的、自适应的流程，将灰度掩模转换为干净的二值掩模，
    并确保SRAF与主图形之间保持最小间距。
    """
    print("\n--- Starting Mask Binarization Process ---")
    
    # --- 阶段 1 & 2: 自适应二值化与伪影滤除
    print("Step 1 & 2: Adaptive Binarization and Artifact Filtering...")
    peak_coords = peak_local_max(gray_mask, min_distance=1, threshold_abs=peak_min_intensity)
    if peak_coords.shape[0] == 0:
        print("Warning: No local peaks found. Returning an empty mask.")
        empty_mask = np.zeros_like(gray_mask, dtype=np.uint8)
        return empty_mask, empty_mask, empty_mask, empty_mask, empty_mask
        
    peak_values = gray_mask[peak_coords[:, 0], peak_coords[:, 1]]
    adaptive_threshold = np.quantile(peak_values, quantile_val)
    initial_binary_mask = (gray_mask >= adaptive_threshold)
    labeled_mask = label(initial_binary_mask)
    props = regionprops(labeled_mask)
    filtered_binary_mask = initial_binary_mask.copy()
    for prop in props:
        if prop.area < area_threshold:
            filtered_binary_mask[labeled_mask == prop.label] = 0
            
    # --- 阶段 3: 提取初步的 SRAF ---
 
    sraf_mask_initial = filtered_binary_mask & (~(target_pattern > 0.5))

    # --- 阶段 4: 强制SRAF间距 ---
    print(f"Step 3: Enforcing {min_spacing_pixels} pixel minimum spacing for SRAFs...")
    if min_spacing_pixels > 0:
        keep_out_zone = binary_dilation(target_pattern > 0.5, footprint=disk(min_spacing_pixels))
        sraf_mask_final = sraf_mask_initial & (~keep_out_zone)
    else:
        sraf_mask_final = sraf_mask_initial
        keep_out_zone = np.zeros_like(target_pattern) # 用于可视化

    
    # --- 阶段 5: 合成版图 ---
    print("Step 4: Composing final mask...")
    final_mask = (target_pattern > 0.5) | sraf_mask_final
    
    return initial_binary_mask, filtered_binary_mask, sraf_mask_final, final_mask.astype(np.uint8), keep_out_zone
