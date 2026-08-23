import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from skimage.measure import label, regionprops
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.morphology import binary_opening, disk
from PIL import Image
from skimage import draw, measure
#可视化版图函数
def show_matrices(
    matrices: Union[np.ndarray, List[np.ndarray]],
    titles: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Optional[tuple] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """
    可视化一个或多个矩阵（并排显示，带 colorbar）

    Parameters
    ----------
    matrices : np.ndarray or List[np.ndarray]
        单个矩阵或矩阵列表
    titles : List[str], optional
        每个矩阵对应的标题
    cmap : str
        颜色映射（默认 viridis，科研常用）
    figsize : tuple, optional
        图像尺寸，例如 (12, 4)
    vmin, vmax : float, optional
        颜色范围，多个矩阵对比时建议设置
    """

    # 统一成 list
    if isinstance(matrices, np.ndarray):
        matrices = [matrices]

    n = len(matrices)

    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(n)]
    else:
        assert len(titles) == n, "titles 数量必须与矩阵数量一致"

    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    # 只有一个子图时，axes 不是 list
    if n == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def _prune_spurs(adj, max_len):
    adj = adj.copy()
    degrees = np.sum(adj, axis=1)

    branch_nodes = np.where(degrees >= 3)[0]

    for b in branch_nodes:
        neighbors = np.where(adj[b])[0]
        for n in neighbors:
            path = [b, n]
            prev, curr = b, n

            for _ in range(max_len):
                deg = np.sum(adj[curr])
                if deg >= 3:
                    break
                nexts = [x for x in np.where(adj[curr])[0] if x != prev]
                if not nexts:
                    break
                prev, curr = curr, nexts[0]
                path.append(curr)

            # 如果是短端点分支，删除
            if np.sum(adj[curr]) == 1 and len(path) <= max_len:
                for i in range(len(path) - 1):
                    adj[path[i], path[i + 1]] = False
                    adj[path[i + 1], path[i]] = False

    return adj

#骨架图上选点函数
def get_ordered_path(pts, spur_max_len=3):
    """
    对骨架点进行排序：
    1) 自动剪掉短毛刺（spur）
    2) 输出一条干净的折线或闭合环
    """

    N = len(pts)
    if N < 3:
        return pts, False

    # -------------------------------------------------
    # 1. 构建 8 邻接图
    # -------------------------------------------------
    dists = cdist(pts, pts)
    adj = (dists <= 1.42) & (dists > 0)

    # -------------------------------------------------
    # 2. 剪毛刺：从分叉点出发，删除短分支
    # -------------------------------------------------
    adj = _prune_spurs(adj, spur_max_len)

    degrees = np.sum(adj, axis=1)

    # -------------------------------------------------
    # 3. 判断是否闭合
    # -------------------------------------------------
    endpoints = np.where(degrees == 1)[0]
    if len(endpoints) > 0:
        start_node = endpoints[0]
        is_closed = False
    else:
        start_node = np.where(degrees > 0)[0][0]
        is_closed = True

    # -------------------------------------------------
    # 4. 线性路径追踪（degree ≤ 2，保证不乱走）
    # -------------------------------------------------
    ordered = [start_node]
    visited = {start_node}

    while True:
        curr = ordered[-1]
        neighbors = np.where(adj[curr])[0]
        next_nodes = [n for n in neighbors if n not in visited]
        if not next_nodes:
            break
        ordered.append(next_nodes[0])
        visited.add(next_nodes[0])

    return pts[ordered], is_closed

#在骨架图上均匀采点函数
def extract_sraf_parameters(skel, spacing=10):
    """
    主函数：提取控制点并识别拓扑结构
    """
    labeled = label(skel, connectivity=2)
    regions = regionprops(labeled)
    sraf_data = [] # 存储字典：{points, is_closed}
    sum_cps = 0
    
    
    for region in regions:
        # region.image = cv2.GaussianBlur(region.image.astype(float),(5,5),1) > 0.5
        skel = region.image
        y, x = np.nonzero(skel)
        pts = np.stack([y + region.bbox[0], x + region.bbox[1]], axis=1)
        
        if len(pts) < 3: continue # 至少3个点才能构成有意义的路径
        
        # 1. 路径排序与拓扑识别
        ordered_pts, is_closed = get_ordered_path(pts)
        total_len = len(ordered_pts)
        sum_cps+= len(ordered_pts)
        # 2. 采样逻辑改进
        if not is_closed:
            # --- 非闭合：确保包含首尾 ---
            if total_len <= 4:
                control_pts = ordered_pts.astype(float)
            else:
                # 计算中间需要多少个点
                num_mid_samples = max(2, int(np.ceil(total_len / spacing)) - 2)
                # 生成从 0 到 total_len-1 的索引，包含两端
                indices = np.linspace(0, total_len - 1, num_mid_samples + 2).astype(int)
                # 去重并保持顺序
                indices = np.unique(indices)
                control_pts = ordered_pts[indices].astype(float)
        else:
            # --- 闭合：均匀采样，不取重复终点 ---
            num_samples = max(4, int(np.ceil(total_len / spacing)))
            indices = np.linspace(0, total_len - 1, num_samples, endpoint=False).astype(int)
            control_pts = ordered_pts[indices].astype(float)
        
        sraf_data.append({
            'control_points': control_pts,
            'is_closed': is_closed
        })
        
    return sraf_data

def get_spline_curve(ctrl_pts, is_closed, n_render=200):
    """
    根据控制点和闭合标志生成平滑的B-Spline曲线
    """
    k = min(3, len(ctrl_pts) - 1) # 样条阶数，至少3阶
    if k < 1: # 至少需要2个点才能有1阶样条
        return ctrl_pts # 退化为直线
    
    try:
        # per=is_closed 是关键参数，用于闭合曲线
        tck, u = splprep([ctrl_pts[:, 0], ctrl_pts[:, 1]], s=0, k=k, per=is_closed)
        u_fine = np.linspace(0, 1, n_render)
        curve = splev(u_fine, tck)
        return np.stack(curve, axis=1)
    except Exception as e:
        print(f"Warning: B-Spline fitting failed for {len(ctrl_pts)} points (k={k}, closed={is_closed}). Error: {e}. Returning control points as path.")
        return ctrl_pts # 拟合失败时，返回控制点连成的折线

def visualize_sraf_parameters(mask, sraf_data, title="SRAF Parameters Visualization"):
    """
    可视化原始SRAF Mask、提取的控制点以及拟合的B-Spline曲线。
    sraf_data: extract_sraf_parameters 函数的输出，一个字典列表
    """
    plt.figure(figsize=(12, 12))
    
    # 1. 绘制原始 Mask 作为背景
    plt.imshow(mask, cmap='gray', interpolation='nearest', alpha=0.5)
    
    # 2. 遍历每一根 SRAF 进行绘制
    for i, data in enumerate(sraf_data):
        ctrl_pts = data['control_points']
        is_closed = data['is_closed']
        
        # 定义颜色和样式
        curve_color = 'limegreen' if is_closed else 'cyan'
        line_style = '-' if is_closed else '--'
        marker_color = 'red' if is_closed else 'blue'
        
        # 2.1 绘制控制点和它们之间的连线
        plt.plot(ctrl_pts[:, 1], ctrl_pts[:, 0], 
                 linestyle=line_style, linewidth=1, color='gray', alpha=0.5, 
                 label=f'SRAF {i+1} Ctrl Pts Path' if i < 10 else None) # 只显示少量图例
        plt.scatter(ctrl_pts[:, 1], ctrl_pts[:, 0], 
                    s=50, c=marker_color, edgecolors='white', zorder=5, 
                    label=f'SRAF {i+1} Control Points ({"Closed" if is_closed else "Open"})' if i < 10 else None)
        
        # 2.2 绘制 B-Spline 拟合曲线
        spline_curve = get_spline_curve(ctrl_pts, is_closed)
        plt.plot(spline_curve[:, 1], spline_curve[:, 0], 
                 linestyle='-', linewidth=2, color=curve_color, zorder=4,
                 label=f'SRAF {i+1} B-Spline Curve' if i < 10 else None)
        
        # 2.3 在起点处标记序号
        plt.text(ctrl_pts[0, 1], ctrl_pts[0, 0], 
                 f"#{i+1}", color='yellow', fontsize=10, fontweight='bold', 
                 ha='right', va='bottom')

    plt.title(title, fontsize=16)
    plt.xlabel("X (pixels)", fontsize=12)
    plt.ylabel("Y (pixels)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='lower right', fontsize='medium')
    plt.axis('equal') # 保持XY轴比例一致，避免变形
    plt.show()
def extract_sraf(full_mask, main_mask, thresh=0.5):

    labels = measure.label(full_mask, connectivity=2)

    main_from_full = np.zeros_like(full_mask, dtype=np.uint8)
    sraf_from_full = np.zeros_like(full_mask, dtype=np.uint8)

    for region in measure.regionprops(labels):

        coords = region.coords
        area = region.area

        overlap = np.sum(main_mask[coords[:,0], coords[:,1]])
        ratio = overlap / area

        if ratio > thresh:
            main_from_full[coords[:,0], coords[:,1]] = 1
        else:
            sraf_from_full[coords[:,0], coords[:,1]] = 1

    return main_from_full, sraf_from_full

from skimage.morphology import medial_axis
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
diagonal_LSM_mask = np.loadtxt(r"outputs/opc/CTM和levelset图像/Ls_mask/ls_image对角通孔pe_epe_pvband无切趾.txt")
contact_LSM_mask = np.loadtxt(r"outputs/opc/CTM和levelset图像/Ls_mask/ls_image中心对称通孔pe_epe_pvband.txt")
# img = Image.open(r".\outputs/opc\DPS\BS\0.7_BS_中心对称通孔\未优化版图_mask.png").convert('L')  # 转灰度
# arr = np.array(img)
# binary = (arr >= 128).astype(np.uint8)
# target_mask2 = binary
target_digonal = np.loadtxt(r"outputs/opc/CTM和levelset图像/目标版图/对角通孔目标版图.txt")

_, diagonal_sraf = extract_sraf(diagonal_LSM_mask,target_digonal)
# _, sraf2 = extract_sraf(image2,target_mask2)
binary_open_diagonal_sraf = binary_opening(diagonal_sraf, disk(2))

    # 2 距离变换
dist_diagonal = distance_transform_edt(binary_open_diagonal_sraf)


skelet_diagonal_sraf = skeletonize(binary_open_diagonal_sraf)
# plt.imshow(cv2.GaussianBlur(sraf1.astype(float),(5,5),1) > 0.5)
# plt.title("高斯模糊之后的sraf1")
# plt.show()
show_matrices([diagonal_LSM_mask,binary_open_diagonal_sraf,dist_diagonal,skelet_diagonal_sraf], 
              titles=["diagonal_LSM_mask","binary_open_diagonal_sraf","dist_digonal_sraf","skelet_diagonal_sraf"])

# show_matrices([gusssraf1, skel1,centerline1], titles=["sraf1", "skel1","centerline1"])
# show_matrices([sraf2, skel2], titles=["sraf2", "skel2"])

dignoal_sraf_params = extract_sraf_parameters(skelet_diagonal_sraf  , spacing=10)
# sraf_params2 = extract_sraf_parameters(skel2 , spacing=10)
visualize_sraf_parameters(skelet_diagonal_sraf,dignoal_sraf_params)
# visualize_sraf_parameters(skel2,sraf_params2)


# show_matrices([image1, image2], titles=["对角通孔", "中心对称通孔"])