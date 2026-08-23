import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import binary_erosion
from typing import Tuple, List


def create_test_pattern(pattern_type: str = 'rect', size: int = 256) -> np.ndarray:
    """生成一个用于测试的图形"""
    if pattern_type == 'rect':
        # 创建一个简单的矩形
        pattern = np.zeros((size, size))
        pattern[size//4 : 3*size//4, size//4 : 3*size//4] = 1.0
        return pattern
    elif pattern_type == 'l-shape':
        # 创建一个L形
        pattern = np.zeros((size, size))
        pattern[size//4 : 3*size//4, size//4 : size//2] = 1.0
        pattern[size//2 : 3*size//4, size//4 : 3*size//4] = 1.0
        return pattern
    else: # 默认圆形
        diameter = size // 2
        pattern = np.zeros((size, size))
        radius = diameter / 2
        center = size / 2
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        distance = np.sqrt((xx - center)**2 + (yy - center)**2)
        pattern[distance <= radius] = 1.0
        return pattern


def estimate_min_feature_size(pixel_mask: np.ndarray) -> float:
    """
    使用形态学腐蚀来估算掩模中的最小特征尺寸（宽度）。
    返回值为像素数。
    """
    if not np.any(pixel_mask):
        return 0.0
    mask_bool = pixel_mask > 0.5
    eroded_mask = mask_bool.copy()
    width_iterations = 0
    while np.any(eroded_mask):
        eroded_mask = binary_erosion(eroded_mask)  #每腐蚀一次就就在左右两侧消去1个像素，那就是消去两个像素，对于奇数就会不准，
        width_iterations += 1
    min_width_estimate = width_iterations * 2
    return max(1.0, float(min_width_estimate))

def get_geometric_key_points_adaptive(
    pixel_mask: np.ndarray,
    contour_level: float = 0.5,
    tolerance_factor: float = 0.25,
    include_corners: bool = True, # Demo中我们同时看角点
    include_line_midpoints: bool = True,
    ensure_points_on_foreground: bool = True
) -> Tuple[List, List]:
    """
    自适应提取几何关键点, 并可选地确保这些点落在pattern上。
    Returns:
    final_line_mid_points_adj (角点没有意义)
    """
    min_cd_pixels = estimate_min_feature_size(pixel_mask)
    adaptive_tolerance = max(1.0, min_cd_pixels * tolerance_factor)
    print(f"估算的最小特征尺寸约为: {min_cd_pixels:.2f} 像素")
    print(f"使用的自适应公差 (tolerance): {adaptive_tolerance:.2f}")
    pixel_mask_bool = (pixel_mask > contour_level)
    pixel_mask_for_value_check = pixel_mask.astype(pixel_mask_bool.dtype)
    contours = find_contours(pixel_mask_bool, level=contour_level)
    all_corner_points_tuples = []
    all_line_mid_points_tuples = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approximated_polygon_vertices = approximate_polygon(contour, tolerance=adaptive_tolerance)  #根据dp算法简化轮廓
        current_contour_geometric_corners = []
        if len(approximated_polygon_vertices) > 0:
            if np.allclose(approximated_polygon_vertices[0], approximated_polygon_vertices[-1]) and len(approximated_polygon_vertices) > 1:
                vertices = approximated_polygon_vertices[:-1]  #如果收尾有重复就去掉尾部的数据点
            else:
                vertices = approximated_polygon_vertices
            current_contour_geometric_corners = [tuple(v.astype(int)) for v in vertices] # 将轮廓点转换为元组列表
        if include_corners:
            all_corner_points_tuples.extend(current_contour_geometric_corners)  # 将当前轮廓的角点添加到所有角点列表中
        if include_line_midpoints and len(current_contour_geometric_corners) >= 2:
            num_edges = len(current_contour_geometric_corners)  # 计算当前轮廓的边数
            for i in range(num_edges):
                p1 = np.array(current_contour_geometric_corners[i])
                p2 = np.array(current_contour_geometric_corners[(i + 1) % num_edges])
                mid_point_y = (p1[0] + p2[0]) / 2.0
                mid_point_x = (p1[1] + p2[1]) / 2.0
                all_line_mid_points_tuples.append((int(round(mid_point_y)), int(round(mid_point_x)))) #将当前轮廓的线段中点添加到所有线段中点列表中
    final_corner_points = sorted(list(set(all_corner_points_tuples)))
    final_line_mid_points = sorted(list(set(all_line_mid_points_tuples)))
    if include_corners and include_line_midpoints:
        corner_set_final = set(final_corner_points)
        final_line_mid_points = [p for p in final_line_mid_points if p not in corner_set_final]
    if not include_corners: final_corner_points = []
    if not include_line_midpoints: final_line_mid_points = []
    if not ensure_points_on_foreground:
        print(f"角点数量: {len(final_corner_points)}")
        print(f"直线边缘中点数量: {len(final_line_mid_points)}")
        return final_corner_points, final_line_mid_points
    adjusted_corner_points = []
    adjusted_line_mid_points = []
    height, width = pixel_mask_for_value_check.shape
    point_lists_to_process = [
        (final_corner_points, adjusted_corner_points),
        (final_line_mid_points, adjusted_line_mid_points)
    ]
    for point_list, adjusted_list in point_lists_to_process:
        for r_orig, c_orig in point_list:
            if not (0 <= r_orig < height and 0 <= c_orig < width):
                print(f"警告: 原始提取点 ({r_orig},{c_orig}) 超出掩模范围，已忽略。")
                continue
            if np.isclose(pixel_mask_for_value_check[r_orig, c_orig], 1.0):
                adjusted_list.append((r_orig, c_orig))
                continue
            found_foreground_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    r_new, c_new = r_orig + dr, c_orig + dc
                    if 0 <= r_new < height and 0 <= c_new < width and \
                       np.isclose(pixel_mask_for_value_check[r_new, c_new], 1.0):
                        adjusted_list.append((r_new, c_new))
                        found_foreground_neighbor = True
                        break
                if found_foreground_neighbor:
                    break
            if not found_foreground_neighbor:
                print(f"警告: 点 ({r_orig},{c_orig}) 及其邻域未找到值为1的像素，已忽略此点。")
    final_corner_points_adj = sorted(list(set(adjusted_corner_points)))
    final_line_mid_points_adj = sorted(list(set(adjusted_line_mid_points)))
    if include_corners and include_line_midpoints:
        corner_set_adj = set(final_corner_points_adj)
        final_line_mid_points_adj = [p for p in final_line_mid_points_adj if p not in corner_set_adj]
    print(f"调整后 - 角点数量: {len(final_corner_points_adj)}")
    print(f"调整后 - 直线边缘中点数量: {len(final_line_mid_points_adj)}")
    return final_corner_points_adj, final_line_mid_points_adj

