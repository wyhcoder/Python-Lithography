import numpy as np
import cv2

from matplotlib import pyplot as plt
from skimage import draw, measure
from scipy.ndimage import  map_coordinates,distance_transform_edt,binary_dilation
from skimage.measure import label
import psutil
import copy
import networkx as nx
from scipy.spatial import cKDTree
from pathlib import Path
from utils_model.demo_MSAA import AntiAliasRenderer
from utils_model.demo_parametric import ParametricDemo
from skimage.morphology import medial_axis
import skimage.measure as skm
from skimage.morphology import dilation, disk, skeletonize
from skimage.morphology import binary_opening, disk
from scipy.interpolate import splprep, splev
# def extract_sraf(full_mask,origin_mask):
#         """用于分离sraf和主图形，只适用于二值版图

#         Args:
#             full_mask (_type_): 要分离的版图
#             origin_mask (_type_): 目标版图：只包含主图形

#         Returns:
#             _type_: _description_
#         """
       
   
#         labels = measure.label(full_mask, connectivity=2)
#         regions = measure.regionprops(labels)

#         main_mask = np.zeros_like(full_mask)
#         sraf_mask = np.zeros_like(full_mask)

#         for region in regions:
#             temp_mask = (labels == region.label).astype(np.uint8)
#             overlap = np.sum(temp_mask * origin_mask)
#             area = np.sum(temp_mask)
#             if overlap / area > 0.6:
#                 main_mask[temp_mask == 1] = 1
#             else:
#                 sraf_mask[temp_mask == 1] = 1

        
#         return main_mask, sraf_mask
def extract_sraf(
    full_mask: np.ndarray,
    origin_mask: np.ndarray,
    dilate_radius: int = 3,
    fg_thresh: float = 1e-6,
    overlap_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 full_mask 中分离主图形和 SRAF，保留原灰度值。

    改进点（相对旧版"仅按 origin 膨胀区域分"）：
      - 对 full_mask 做前景连通域标记 (8 邻接)；
      - 每个连通域按"与 origin_mask 的真重叠"判断归属：
          * 重叠 >= max(1px, overlap_ratio * 域面积) 视为主图形；
          * 否则视为 SRAF。
      - 旧版的"膨胀 origin 当主图形区域"只用作回退/兜底，
        不再决定边界归属，避免主图形偏移/形变后边缘被错判为 SRAF。

    参数:
        full_mask:     包含主图形和 SRAF 的完整掩模 (可为灰度)。
        origin_mask:   只含主图形的参考掩模 (>0 视为前景)。
        dilate_radius: 旧接口兼容参数；当 origin 与 full_mask 主图形
                       几乎完全无重叠时，会用此半径膨胀 origin 来兜底。
        fg_thresh:     前景判定阈值（灰度像素 >fg_thresh 视为前景）。
        overlap_ratio: 一个连通域被判为主图形的最小重叠占比。

    返回:
        main_mask, sraf_mask : 与 full_mask 同 dtype，同灰度值。
    """
    full_mask = np.asarray(full_mask)
    full_f = full_mask.astype(np.float32, copy=False)
    fg = (full_f > fg_thresh).astype(np.uint8)

    origin_bin = (np.asarray(origin_mask) > 0).astype(np.uint8)

    main_mask = np.zeros_like(full_mask)
    sraf_mask = np.zeros_like(full_mask)

    if fg.sum() == 0:
        return main_mask, sraf_mask

    # 1) 连通域标记（8 邻接）+ 按"真重叠面积"决定归属
    num, labels = cv2.connectedComponents(fg, connectivity=8)
    matched_any = False
    for lbl in range(1, num):
        comp = (labels == lbl)
        area = int(comp.sum())
        if area == 0:
            continue
        overlap = int(np.logical_and(comp, origin_bin).sum())
        # 主图形判据：与 origin 真实有像素重叠，且占比 >= overlap_ratio
        is_main = (overlap > 0) and (overlap >= max(1, int(overlap_ratio * area)))
        if is_main:
            main_mask[comp] = full_mask[comp]
            matched_any = True
        else:
            sraf_mask[comp] = full_mask[comp]

    # 2) 兜底：若连通域分析没能匹配到任何主图形（origin 与 full_mask 完全
    #    不重叠的极端情况），退化为旧版逻辑"膨胀 origin 当主图形区域"。
    if not matched_any:
        dilated_origin = binary_dilation(origin_bin, structure=disk(dilate_radius))
        main_mask[:] = 0
        sraf_mask[:] = 0
        main_mask[dilated_origin] = full_mask[dilated_origin]
        sraf_mask[~dilated_origin] = full_mask[~dilated_origin]

    return main_mask, sraf_mask

def visualize_srafs_by_order(SRAFs_layer, file_path, title="SRAFs by Order"):
        """
        可视化不同阶的 SRAF, 使用柔和配色 + 清晰图例.

        SRAFs_layer: dict, key=order, value=2D bool mask
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        orders = sorted(SRAFs_layer.keys())
        if not orders:
            print("SRAFs_layer is empty!")
            return
        H, W = next(iter(SRAFs_layer.values())).shape

        # 把每阶 SRAF 写到统一的 order_map 里
        order_map = np.zeros((H, W), dtype=int)
        for order in orders:
            mask = SRAFs_layer[order].astype(bool)
            order_map[mask] = order

        # 柔和配色: 浅灰背景 + 4 阶按"近主图形->远"递变, 接近 ColorBrewer
        # 0=背景(浅灰); 1=Order1(深蓝); 2=Order2(青); 3=Order3(橙); 4=Order4(粉红)
        color_list = [
            "#F2F2F2",  # 0 background
            "#1f77b4",  # 1 Order 1 (closest to main)
            "#17becf",  # 2 Order 2
            "#ff7f0e",  # 3 Order 3
            "#e377c2",  # 4 Order 4 (farthest)
        ]
        # 若有更多阶, 自动循环
        while len(color_list) < max(orders) + 1:
            color_list.append("#9467bd")
        cmap = mcolors.ListedColormap(color_list[:max(orders) + 1])
        norm = mcolors.BoundaryNorm(
            boundaries=np.arange(-0.5, max(orders) + 1),
            ncolors=max(orders) + 1,
        )

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.imshow(order_map, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_axis_off()

        # 简洁图例 (右下角, 半透明白底, 黑细边)
        legend_handles = [
            Patch(facecolor=color_list[o], edgecolor="black", linewidth=0.6,
                  label=f"Order {o}")
            for o in orders
        ]
        leg = ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=True,
            framealpha=0.95,
            edgecolor="black",
            fontsize=10,
        )
        leg.get_frame().set_linewidth(0.6)

        # 紧凑保存, 无白边
        save_path = f"{file_path}/order_map.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return order_map
        
 
def extract_sraf_by_order_fixed(full_mask, target_mask, max_order=4, search_dist=20):
            """
            按阶数提取SRAF：基于连通域生长，解决跨距识别和图形破碎问题
            return: sraf_list
            """
            # 1. 提取纯SRAF区域（调用你之前定义的 overlap 判定函数）
            _, sraf_only = extract_sraf(full_mask, target_mask)
            
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


def classify_sraf_byDistance(
            main_mask,
            sraf_mask,
            filepath,
            num_orders=4,
            dist_tol=2.0,
            use_median=True
        ):
            """
            将 SRAF 按到主图形的距离分阶：
            - 距离接近的归为同一阶
            - 只保留离主图形最近的前 num_orders 阶
            - 不强制并阶，不把远处 SRAF 合并到最后一阶
            """

            # --------------------------------------------------
            # 0. 基本检查 + dtype 统一
            # --------------------------------------------------
            assert main_mask.shape == sraf_mask.shape

            main_mask = main_mask.astype(bool)
            sraf_mask = sraf_mask.astype(bool)
            main_skeleton = skeletonize(main_mask)
            
            # 对于SRAF先进行形态学开操作在进行骨架提取
            sraf_mask = binary_opening(sraf_mask,disk(2))
            skelet = skeletonize(sraf_mask)
            
            # --------------------------------------------------
            # 1. 连通域标记（8 连通）
            # --------------------------------------------------
            labeled_sraf, num_sraf = label(
                sraf_mask,
                connectivity=2,
                return_num=True
            )


            if num_sraf == 0:
                return {}, np.zeros_like(sraf_mask, dtype=np.uint8), {}

            # --------------------------------------------------
            # 2. 距离场（到主图形）
            # --------------------------------------------------
            dist_map = distance_transform_edt(~main_skeleton)
            
            # sraf_dist SRAF到主图形骨架图的距离，sraf_centers：SRAF的中心点
            sraf_dist = {}
            sraf_centers = {}
            for lb in range(1, num_sraf + 1):
                # 找到当前 label 对应的骨架点
                # (labeled_sraf == lb) 确定是哪一个 SRAF
                # skel_indice 确定哪些是骨架点
                skel_indices = np.where((labeled_sraf == lb) & skelet)
                
                if skel_indices[0].size == 0:
                    # 如果 SRAF 太小无法形成骨架，退化回使用该连通域的所有点
                    pts = np.where(labeled_sraf == lb)
                    d = dist_map[pts]
                    sraf_dist[lb] = np.round(np.median(d), 2)
                    # 取重心作为文字位置
                    sraf_centers[lb] = (np.mean(pts[0]), np.mean(pts[1]))
                    continue

                # 获取骨架线上各点到主图形的距离
                skel_dists = dist_map[skel_indices]

                # 取骨架距离的中位数作为该 SRAF 的代表距离
                # 这样即使 SRAF 是弯曲的，也能得到一个稳定的“中心轨道”距离
                sraf_dist[lb] = np.median(skel_dists)
                # 取骨架线的中点位置作为文字标注点
                mid_idx = len(skel_indices[0]) // 2
                sraf_centers[lb] = (skel_indices[0][mid_idx], skel_indices[1][mid_idx])
            
            # --------------------------------------------------
            # 4. 按距离排序
            # --------------------------------------------------
            items = sorted(sraf_dist.items(), key=lambda x: x[1]) # lambda是匿名函数，按照x[1]排序

            # --------------------------------------------------
            # 5. 距离容差分阶（不强制并阶，只取前 num_orders）
            # --------------------------------------------------
            sraf_by_order = {}
            skelet_by_order = {}
            current_order = 1
            current_group = []

            for label_id, dist in items:
                if not current_group:
                    current_group = [(label_id, dist)]
                    continue

                prev_dist = current_group[-1][1]  # 取当前组最后一个元素的dist,上一个的dist

                if abs(dist - prev_dist) <= dist_tol:
                    current_group.append((label_id, dist))
                else:
                    # 结束一阶
                    mask = np.zeros_like(sraf_mask, dtype=bool)
                    skelet_mask = np.zeros_like(sraf_mask, dtype=bool)
                    for lb, _ in current_group:
                        mask[labeled_sraf == lb] = True
                        skelet_mask[(labeled_sraf == lb) & skelet]  = True
                    
                    sraf_by_order[current_order] = mask
                    skelet_by_order[current_order] = skelet_mask

                    current_order += 1
                    if current_order > num_orders:
                        break

                    current_group = [(label_id, dist)]

            # --------------------------------------------------
            # 6. 补上最后一阶
            # --------------------------------------------------
            if current_group and current_order <= num_orders:
                mask = np.zeros_like(sraf_mask, dtype=bool)
                skelet_mask = np.zeros_like(sraf_mask, dtype=bool)
                for lb, _ in current_group:
                    mask[labeled_sraf == lb] = True
                    skelet_mask[(labeled_sraf == lb) & skelet]  = True
                
                sraf_by_order[current_order] = mask
                skelet_by_order[current_order] = skelet_mask

            # --------------------------------------------------
            # 7. 合并所有阶的 SRAF（不使用位运算）
            # --------------------------------------------------
            all_sraf = np.zeros_like(sraf_mask, dtype=bool)
            all_skelet = np.zeros_like(sraf_mask, dtype=bool)
            for mask in sraf_by_order.values():
                all_sraf[mask] = True
            for mask in skelet_by_order.values():
                all_skelet[mask] = True
            

            num_pointsOfskeleton  = np.sum(all_skelet)           
            all_sraf_int = all_sraf.astype(np.uint8)
            all_skeleton_int = all_skelet.astype(np.uint8)

            # --------------------------------------------------
            # 8. debug 信息
            # --------------------------------------------------
            debug_info = {
                "labeled_sraf": labeled_sraf,
                "dist_map": dist_map,
                "sraf_dist": sraf_dist,
                "sraf_centers": sraf_centers, # 新增
                "skeleton_all": skelet, # 新增
                "num_sraf": num_sraf,
                "num_orders_used": len(sraf_by_order)
            }
            plot_sraf_distances(debug_info, save_path=f"{filepath}/sraf_distances.png")
            return sraf_by_order, all_sraf_int, skelet_by_order,all_skeleton_int,num_pointsOfskeleton,debug_info


def plot_sraf_distances(debug_info, save_path=None,
                        annotate=True, dedupe_tol=0.3,
                        leader_max_len_frac=0.25):
    """
    在骨架图上标注每个 SRAF 的距离 (到主图骨架的距离).

    设计要点:
    - 浅灰底 + 黑色细骨架, 视觉干净;
    - 数值保留 1 位小数, 圆角胶囊型 label, 字号 9, 不再像原版那样压住骨架;
    - 同一阶往往多条 SRAF 距离值几乎相同 (例如 35.13, 35.13, 35.15),
      用 dedupe_tol 合并 (默认半像素), 同距离的多个点用一根引线指向公共标签,
      彻底消除\"4 个 35.13 挤一团\"的问题;
    - 引线极淡, 不抢视觉焦点.

    Parameters
    ----------
    debug_info : dict
        必须含 'skeleton_all', 'sraf_dist', 'sraf_centers'.
    save_path : str, optional
        若给出, 保存到该路径.
    annotate : bool
        False 时只画骨架, 不放距离标签 (用于纯骨架预览).
    dedupe_tol : float
        距离去重容忍度 (像素). 距离差 <= 该值的 SRAF 共享一个标签.
    leader_max_len_frac : float
        引线最大长度上限 (相对图像短边的比例). 超过该长度的连线不画,
        防止远距离 SRAF 被强行连到一起破坏布局.
    """
    import matplotlib.pyplot as plt

    skeleton = debug_info["skeleton_all"]
    dist_dict = debug_info["sraf_dist"]
    centers = debug_info["sraf_centers"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 浅灰底 + 黑色骨架 (gray_r: 0=白, 1=黑)
    ax.imshow(skeleton, cmap="gray_r", interpolation="nearest",
              vmin=0, vmax=1)
    ax.set_axis_off()

    if annotate and dist_dict:
        H, W = skeleton.shape
        leader_max = leader_max_len_frac * min(H, W)

        # ---- 1. 把所有 (label, dist, y, x) 按 dist 做近似分组 ----
        items = []
        for lb, dist in dist_dict.items():
            y, x = centers[lb]
            items.append((lb, float(dist), float(y), float(x)))
        items.sort(key=lambda t: t[1])

        groups = []  # 每个 group: {dist: 代表值, members: [(y, x), ...]}
        for lb, dist, y, x in items:
            # 同时检查\"距离接近\" + \"空间不太远\" 才合并到上一组,
            # 避免左右两组对称结构被合并成一组导致超长引线.
            merged = False
            if groups and abs(dist - groups[-1]["dist"]) <= dedupe_tol:
                # 计算到上一组任意成员的最近距离
                last_members = groups[-1]["members"]
                min_d = min(((y - my) ** 2 + (x - mx) ** 2) ** 0.5
                            for my, mx in last_members)
                if min_d <= leader_max:
                    last_members.append((y, x))
                    n = len(last_members)
                    groups[-1]["dist"] = (
                        groups[-1]["dist"] * (n - 1) + dist
                    ) / n
                    merged = True
            if not merged:
                groups.append({"dist": dist, "members": [(y, x)]})

        # ---- 2. 为每组选 anchor + 画引线 ----
        cy, cx = H / 2.0, W / 2.0
        for g in groups:
            members = g["members"]
            # 标签锚点放\"离图像中心最远\"的成员处, 同距离多 SRAF 不挤在版图正中
            members.sort(key=lambda p: -((p[0] - cy) ** 2 + (p[1] - cx) ** 2))
            anchor_y, anchor_x = members[0]

            # >=2 个成员时画引线 (淡灰短线)
            for (oy, ox) in members[1:]:
                ax.plot([anchor_x, ox], [anchor_y, oy],
                        color="#888", linewidth=0.5, alpha=0.45,
                        zorder=2)

            ax.text(anchor_x, anchor_y, f"{g['dist']:.1f}",
                    color="#b30000", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#fff7e6", alpha=0.92,
                              edgecolor="#d9a679", linewidth=0.5))

    plt.tight_layout(pad=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
        print(f"图像已保存至: {save_path}")
    plt.close(fig)



def _prune_graph(G, max_len):
    """
    稳健剪枝算法：
    1. 寻找图中所有的度数为1的端点。
    2. 从端点开始向内追踪路径，直到遇到度数 > 2 的分叉点。
    3. 如果该分支路径的节点数 <= max_len，则认为它是毛刺，从图中删除。
    4. 重复此过程，直到没有符合条件的毛刺。
    """
    G = G.copy()  # 不修改原始图
    
    while True:
        changed = False
        nodes_to_remove = set()
        
        # 1. 找到当前图中所有的叶子节点（端点）
        endpoints = [n for n, d in G.degree() if d == 1]
        
        for ep in endpoints:
            # 2. 追踪该端点形成的分支
            branch_path = [ep]
            curr = ep
            
            while True:
                # 寻找下一个邻居
                neighbors = list(G.neighbors(curr))
                # 排除已经已经在路径里的点，找到前进方向
                next_nodes = [n for n in neighbors if n not in branch_path]
                
                if not next_nodes:
                    # 这是一个孤立的小线段（两头都是端点）
                    break
                
                next_node = next_nodes[0]
                
                # 如果遇到了分叉点（度数 > 2），追踪停止
                if G.degree(next_node) > 2:
                    break
                
                # 如果遇到了另一个端点（度数 = 1），说明这是一条独立短线
                if G.degree(next_node) == 1:
                    branch_path.append(next_node)
                    break
                
                # 否则继续沿着度数为 2 的路径向内走
                branch_path.append(next_node)
                curr = next_node
            
            # 3. 如果分支长度小于等于阈值，标记删除
            if len(branch_path) <= max_len:
                nodes_to_remove.update(branch_path)
                changed = True
        
        # 4. 执行删除操作
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
        
        # 如果这一轮没有任何点被删除，说明剪枝完成
        if not changed:
            break
            
    return G



import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

def get_ordered_path(pts, spur_max_len=3):
    """
    使用图论方法对骨架点进行排序 (修正版：解决 DFS 回溯跳跃问题)
    """
    N = len(pts)
    if N < 2: return pts, False
    if N == 2: return pts, False

    # 1. 构建图：寻找 8 邻域 (r=1.5 覆盖对角线)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.5)
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(pairs)

    # 2. 剪枝 (假设你外部有这个函数，或者用 networkx 简单实现)
    # G = _prune_graph(G, spur_max_len) # 暂时注释，如果你有定义请取消注释
    
    if G.number_of_nodes() == 0:
        return pts, False

    # 3. 提取最大连通分量 (去除噪点)
    if nx.number_connected_components(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # 4. 确定起点
    degrees = dict(G.degree())
    # 找到所有端点 (度为 1 的点)
    endpoints = [n for n, d in degrees.items() if d == 1]
    
    is_closed = False
    start_node = list(G.nodes())[0] # 默认随便找一个

    if len(endpoints) >= 2:
        # --- 情况 A: 开放路径 ---
        # 选取其中一个端点作为起点
        # 为了保证方向一致性，可以选坐标 (y, x) 最小的那个端点
        # endpoints.sort(key=lambda idx: (pts[idx][0], pts[idx][1])) 
        start_node = endpoints[0]
        is_closed = False
    elif len(endpoints) == 0:
        # --- 情况 B: 闭合环路 ---
        is_closed = True
        # 闭合环路随便选一个点开始即可
        start_node = list(G.nodes())[0]
    else:
        # 极少见情况：只有一个端点（类似于 '6' 字形），当作开放处理
        start_node = endpoints[0]
        is_closed = False

    # 5. 核心修改：使用贪婪遍历替代 DFS
    # 逻辑：从当前点出发，找一个“没去过”的邻居，一直走下去
    path_indices = [start_node]
    visited = {start_node}
    
    curr = start_node
    while True:
        # 获取当前点的所有邻居
        neighbors = list(G.neighbors(curr))
        
        # 找到所有未访问的邻居
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if not unvisited_neighbors:
            break # 无路可走，结束
            
        # 如果有多个邻居（分叉），这里简单地取第一个
        # 对于骨架线，通常只有一个未访问邻居
        next_node = unvisited_neighbors[0]
        
        path_indices.append(next_node)
        visited.add(next_node)
        curr = next_node

    # 6. 闭合曲线的特殊处理
    # 贪婪遍历在闭合环路时，最后首尾可能没连上，这里不需要额外操作
    # 因为 is_closed 标记会告诉后面的 B-spline 拟合函数去闭合它

    return pts[path_indices], is_closed


def classify_sraf_and_get_cps(
        main_mask,
        sraf_mask,
        num_orders=4,
        dist_tol=2.0,
        sample_spacing=10  # 新增参数：控制点采样间距
    ):
        """
        将 SRAF 按到主图形的距离分阶，并直接提取每一阶中所有 SRAF 的 B-Spline 控制点。
        
        Returns:
            sraf_cps_by_order (dict): { 
                1: [ {points: arr, is_closed: bool}, {points: arr, is_closed: bool}, ... ],
                2: [ ... ]
            }
            debug_masks (dict): 用于调试的中间结果 (sraf_by_order, etc.)
        """
        # --------------------------------------------------
        # 0. 基本预处理
        # --------------------------------------------------
        assert main_mask.shape == sraf_mask.shape
        main_mask = main_mask.astype(bool)
        sraf_mask = sraf_mask.astype(bool)
        
        # 提取主图形骨架（用于计算距离）
        main_skeleton = skeletonize(main_mask)
        num_cps = 0
        
        # SRAF 预处理：开运算去噪 -> 骨架提取
        sraf_mask = binary_opening(sraf_mask, disk(2))
        sraf_skeleton = skeletonize(sraf_mask)
        

        # --------------------------------------------------
        # 1. 连通域标记 & 距离场计算
        # --------------------------------------------------
        labeled_sraf, num_sraf = label(sraf_mask, connectivity=2, return_num=True)
        
        if num_sraf == 0:
            return {}, {}, {}

        # 计算到主图形骨架的距离场
        dist_map = distance_transform_edt(~main_skeleton)

        # --------------------------------------------------
        # 2. 计算每个 SRAF 的代表距离
        # --------------------------------------------------
        sraf_info_list = [] # 存储 (label_id, distance)

        for lb in range(1, num_sraf + 1):
            # 找到该 SRAF 的骨架像素位置
            skel_indices = np.where((labeled_sraf == lb) & sraf_skeleton)
            
            # 如果骨架太小（比如只是一个点或消失了），退回使用整个形状
            if skel_indices[0].size == 0:
                pts = np.where(labeled_sraf == lb)
                if pts[0].size == 0: continue
                rep_dist = np.median(dist_map[pts])
            else:
                rep_dist = np.median(dist_map[skel_indices])
            
            sraf_info_list.append((lb, rep_dist))

        # --------------------------------------------------
        # 3. 按距离排序
        # --------------------------------------------------
        # 按距离从小到大排序
        sorted_srafs = sorted(sraf_info_list, key=lambda x: x[1])

        # --------------------------------------------------
        # 4. 分阶并提取控制点 (核心逻辑)
        # --------------------------------------------------
        sraf_cps_by_order = {}      # 结果字典：{order: [sraf_data1, sraf_data2...]}
        sraf_mask_by_order = {}     # 调试用：每一阶的 Mask
        
        current_order = 1
        current_group_labels = []   # 当前阶包含的 label ID 列表

        # 为了方便处理最后一组，我们手动追加一个哨兵或者在循环后处理
        # 这里采用遍历 sorted_srafs 的方式
        
        for i, (lb, dist) in enumerate(sorted_srafs):
            # 初始化第一组
            if not current_group_labels:
                current_group_labels.append((lb, dist))
                continue
            
            prev_dist = current_group_labels[-1][1]
            
            # 判断是否属于同一阶
            if abs(dist - prev_dist) <= dist_tol:
                current_group_labels.append((lb, dist))
            else:
                # --- 结算上一阶 ---
                # 1. 提取当前组所有 SRAF 的控制点
                cps_list,num_cp = _batch_extract_cps(
                    current_group_labels, labeled_sraf, sraf_skeleton, sample_spacing
                )
                num_cps += num_cp
                sraf_cps_by_order[current_order] = cps_list
                
                # 2. 生成当前阶的 Mask (可选，用于调试/可视化)
                mask_tmp = np.zeros_like(sraf_mask, dtype=bool)
                for l_id, _ in current_group_labels:
                    mask_tmp[labeled_sraf == l_id] = True
                sraf_mask_by_order[current_order] = mask_tmp

                # 3. 准备下一阶
                current_order += 1
                if current_order > num_orders:
                    current_group_labels = [] # 清空，不再处理后续
                    break
                
                current_group_labels = [(lb, dist)]

        # --- 处理最后一阶 (Loop 结束后的剩余部分) ---
        if current_group_labels and current_order <= num_orders:
            cps_list,num_cp =_batch_extract_cps(
                current_group_labels, labeled_sraf, sraf_skeleton, sample_spacing
            )
            num_cps += num_cp
            sraf_cps_by_order[current_order] = cps_list
            
            mask_tmp = np.zeros_like(sraf_mask, dtype=bool)
            for l_id, _ in current_group_labels:
                mask_tmp[labeled_sraf == l_id] = True
            sraf_mask_by_order[current_order] = mask_tmp
        
        all_sraf = np.zeros_like(sraf_mask, dtype=bool)
            
        for mask in sraf_mask_by_order.values():
            all_sraf[mask] = True
        num_skele = np.sum(all_sraf)              
        all_sraf_int = all_sraf.astype(np.uint8)
            
        # --------------------------------------------------
        # 5. 返回结果
        # --------------------------------------------------
        debug_info = {
            "sraf_mask_by_order": sraf_mask_by_order,#字典，每个key是order，value是mask
            "all_order_sraf": all_sraf_int,  #所有order的mask
            "dist_map": dist_map,  
            "sorted_info": sorted_srafs,
            "num_cps": num_cps,
            "num_skeletons":num_skele
        }
        
        return sraf_cps_by_order, debug_info

def _batch_extract_cps(group_labels, labeled_img, skeleton_img, spacing):
        """
        辅助函数：批量提取一组 Label 的控制点
        """
        cps_result = []
        num_cps = 0
        for lb, _ in group_labels:
            # 提取单个 SRAF 的骨架
            # 注意：必须只提取当前 label 的骨架
            single_skel = (labeled_img == lb) & skeleton_img
            
            
            result = _extract_single_trace(single_skel, spacing)
            if result is None:
                continue
            sraf_data, num_sraf_data = result
            cps_result.append(sraf_data)
            num_cps += num_sraf_data
        return cps_result, num_cps

def _extract_single_trace(skel_mask, spacing):
        """
        从单个 SRAF 的骨架 Mask 中提取有序控制点
        """
        y, x = np.nonzero(skel_mask)
        if len(y) < 3: return None
        
        pts = np.stack([y, x], axis=1)
        
        # 1. 路径排序 (使用之前提到的逻辑)
        ordered_pts, is_closed = get_ordered_path(pts) # 需确保此函数可用
        total_len = len(ordered_pts)
        
        # 2. 采样逻辑
        if not is_closed:
            if total_len <= 4:
                control_pts = ordered_pts.astype(float)
            else:
                num_mid_samples = max(2, int(np.ceil(total_len / spacing)) - 2)
                indices = np.linspace(0, total_len - 1, num_mid_samples + 2).astype(int)
                indices = np.unique(indices)
                control_pts = ordered_pts[indices].astype(float)
        else:
            num_samples = max(4, int(np.ceil(total_len / spacing)))
            indices = np.linspace(0, total_len - 1, num_samples, endpoint=False).astype(int)
            control_pts = ordered_pts[indices].astype(float)
        num_cps = len(control_pts)
        return {
            'control_points': control_pts,
            'is_closed': is_closed
        },num_cps


def get_b_spline_points_by_order(sraf_cps_by_order, num_samples=200):
    """
    输入: sraf_cps_by_order (由控制点组成的嵌套字典)
    输出: fitted_curves_by_order
         结构: { 
            阶数: [ (200, 2)的数组, (200, 2)的数组, ... ],
            ... 
         }
    """
    fitted_curves_by_order = {}

    for order_id, sraf_list in sraf_cps_by_order.items():
        order_fitted_points = []
        
        for sraf_item in sraf_list:
            # 1. 提取控制点和闭合标志
            cps = sraf_item['control_points']  # 形状 (N, 2)
            is_closed = sraf_item['is_closed'] # 布尔值
            
            # 2. 健壮性检查：点数太少无法拟合样条时做线性插值
            if len(cps) < 2:
                # 只有一个点时，复制200次
                pts = np.tile(cps, (num_samples, 1))
                order_fitted_points.append(pts)
                continue
            
            try:
                # 3. 执行 B 样条拟合 (splprep)
                # k 为样条阶数，通常设为 3 (三次样条)，如果点数不够则减小
                k_order = min(3, len(cps) - 1)
                
                # y, x 分别是 pts[:, 0] 和 pts[:, 1]
                # s=0 保证曲线必须经过每一个控制点
                # per=is_closed 处理闭合曲线的首尾平滑衔接
                tck, u = splprep([cps[:, 0], cps[:, 1]], s=0, k=k_order, per=int(is_closed))
                
                # 4. 在 [0, 1] 范围内均匀采样 200 个点
                u_new = np.linspace(0, 1, num_samples)
                y_new, x_new = splev(u_new, tck)
                
                # 组合成 (200, 2) 的数组 [y, x]
                fitted_pts = np.stack([y_new, x_new], axis=1)
                order_fitted_points.append(fitted_pts)
                
            except Exception as e:
                # 如果拟合出错（如重复点），退回到线性插值
                print(f"Warning: Order {order_id} fitting error, falling back to linear: {e}")
                u_linear = np.linspace(0, 1, num_samples)
                # 使用线性插值保证依然输出 200 个点
                # 简单的线性路径近似
                indices = np.linspace(0, len(cps)-1, num_samples)
                y_lin = np.interp(indices, np.arange(len(cps)), cps[:, 0])
                x_lin = np.interp(indices, np.arange(len(cps)), cps[:, 1])
                order_fitted_points.append(np.stack([y_lin, x_lin], axis=1))

        fitted_curves_by_order[order_id] = order_fitted_points

    return fitted_curves_by_order

def debug_plot_fitted_curves(fitted_dict, original_mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(original_mask, cmap='gray', alpha=0.3)
    
    colors = ['r', 'g', 'b', 'y', 'm']
    for order_id, curves in fitted_dict.items():
        c = colors[(order_id-1) % len(colors)]
        for curve in curves:
            # 绘制 200 个采样点连成的线
            plt.plot(curve[:, 1], curve[:, 0], color=c, linewidth=1)
            # 绘制首尾点，检查闭合
            plt.scatter(curve[0, 1], curve[0, 0], color='white', s=5, zorder=5)
            
    plt.title("Fitted 200-pt B-Spline Curves by Order")
    plt.show()
def visualize_sraf_and_controls(mask, sraf_cps_by_order, fitted_curves_by_order,
                                filepath, num_CPS, num_skele,
                                title="SRAF Control Points & B-Spline Curve"):
    """
    可视化 SRAF 多阶分层结果 + 控制点 + B-spline 曲线.

    设计要点:
    - 主图浅灰底色, 突出曲线;
    - 4 阶用与 order_map 一致的柔和配色, 视觉上前后呼应;
    - 控制点连线极淡, 拟合曲线适中粗细, 控制点小圆点带白色描边;
    - 标签只在每阶的第一条 SRAF 上出现, 避免画面拥挤;
    - 统计框放右上, 图例置于图外底部一行, 互不遮挡.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. 底图: 浅灰背景, mask 区域稍亮 (淡化处理, 让控制点和曲线更突出)
    ax.imshow(mask, cmap="Greys", interpolation="nearest", alpha=0.18, vmin=0, vmax=1)

    # 2. 与 order_map 一致的柔和配色
    palette = {
        1: "#1f77b4",   # 深蓝 (Order 1)
        2: "#17becf",   # 青 (Order 2)
        3: "#ff7f0e",   # 橙 (Order 3)
        4: "#e377c2",   # 粉红 (Order 4)
    }
    fallback_colors = ["#9467bd", "#bcbd22", "#8c564b"]

    legend_items = []  # [(label, line_or_patch), ...]

    for i, (order_idx, sraf_list) in enumerate(sraf_cps_by_order.items()):
        order_color = palette.get(order_idx,
                                  fallback_colors[i % len(fallback_colors)])

        for j, sraf_item in enumerate(sraf_list):
            ctrl_pts = sraf_item["control_points"]
            is_closed = sraf_item["is_closed"]

            # 2.1 控制点连线 (极淡的虚线作为辅助参考)
            plot_pts = ctrl_pts
            if is_closed and len(ctrl_pts) > 2:
                plot_pts = np.vstack([ctrl_pts, ctrl_pts[0]])
            ax.plot(plot_pts[:, 1], plot_pts[:, 0],
                    linestyle="--", linewidth=0.5, color="0.6", alpha=0.5,
                    zorder=2)

            # 2.2 拟合曲线 (实心, 中等粗细)
            if fitted_curves_by_order and order_idx in fitted_curves_by_order:
                curve = fitted_curves_by_order[order_idx][j]
                ax.plot(curve[:, 1], curve[:, 0],
                        linestyle="-", linewidth=1.6, color=order_color,
                        alpha=0.9, zorder=4,
                        solid_capstyle="round")

            # 2.3 控制点 (小圆点 + 白描边, 突出且不遮挡)
            ax.scatter(ctrl_pts[:, 1], ctrl_pts[:, 0],
                       s=18, c=order_color,
                       edgecolors="white", linewidths=0.6,
                       zorder=5)

            # 2.4 仅在每阶第一条 SRAF 上贴一个简洁标签 (其他不标)
            if j == 0:
                lx, ly = ctrl_pts[0, 1], ctrl_pts[0, 0]
                ax.text(lx, ly, f"O{order_idx}",
                        color="white", fontsize=9, fontweight="bold",
                        ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor=order_color, alpha=0.85,
                                  edgecolor="white", linewidth=0.5))

        # 收集图例条目
        legend_items.append((f"Order {order_idx}", order_color))

    # 3. 统计文本框 (右上角)
    stats_text = (f"Total CPs: {num_CPS}\n"
                  f"Skeleton Pixels: {num_skele}")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, color="#222",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", alpha=0.9,
                      edgecolor="0.5", linewidth=0.6))

    # 4. 标题 + 坐标轴
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(0, mask.shape[1])
    ax.set_ylim(mask.shape[0], 0)  # imshow 默认 y 轴向下
    ax.set_aspect("equal")
    ax.set_axis_off()

    # 5. 图例放图外底部一行 (代理对象, 一个图例条目同时显示线 + 点)
    handles = []
    for label, color in legend_items:
        h = Line2D([0], [0],
                   color=color, linewidth=2,
                   marker="o", markersize=6,
                   markerfacecolor=color,
                   markeredgecolor="white", markeredgewidth=0.6,
                   label=label)
        handles.append(h)
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=len(handles),
                    frameon=True, framealpha=0.95,
                    edgecolor="0.5", fontsize=10)
    leg.get_frame().set_linewidth(0.6)

    save_path = f"{filepath}/参数化曲线拟合后的曲线以及控制点示意.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"图像已保存至: {save_path}")
    plt.close(fig)















#1. MSAA 子像素 offsets ,用于生成子像素点
def generate_sample_offsets( samples=4):
        offsets = np.linspace(
            -(samples - 1) / (2 * samples),
            (samples - 1) / (2 * samples),
            samples
        )
        oy, ox = np.meshgrid(offsets, offsets)
        return np.stack([oy.ravel(), ox.ravel()], axis=1) 
        
# 2. 距离阈值判定（向量化 + batch）
# =========================================================
def vectorized_is_within_width(
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
def rasterize_distance_msaa_single(
        sraf_mask,
        half_width,
        sample_offsets,
        batch_size=None,
        max_mem_fraction=0.2
    ):
        H, W = sraf_mask.shape

        # --- 中轴线 ---#原本采用中轴线现在采用骨架图
        # centerline, _ = medial_axis(sraf_mask, return_distance=True)
        # mask_fat = dilation(sraf_mask, disk(1))
        # mask_smooth = cv2.GaussianBlur(sraf_mask.astype(float),(5,5),1) > 0.5
        skel = skeletonize(sraf_mask)
        
        # plt.figure(figsize=(6, 6))
        # plt.imshow(sraf_mask, cmap='gray')
        # plt.imshow(skel, cmap='hot', alpha=0.7)  # 中轴叠加
        # plt.axis('off')
        # plt.title('SRAF Mask with Medial Axis')
        # plt.show()

        # --- 到骨架线的距离 ---
        dist_map = distance_transform_edt(~skel )

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

        inside_sub = vectorized_is_within_width(
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
        full_sraf_mask,
        sraf_width_array,
        sraf_layers,
        samples=4,
        show=False
    ):
        #full_sraf_mask: 初始的分阶后的sraf版图
        #sraf_width_array: 分阶后的sraf宽度，有可能是1，也有可以是分的阶数
        #target: 目标图像
        #samples: 每个像素的采样点数
        
        # ===== 强制 SRAF width 扁平化 & 标量化 =====
        sraf_width_array = np.asarray(sraf_width_array)

        if sraf_width_array.ndim > 1:
            # 比如 [[w1], [w2], [w3]] 或 [array([w1]), array([w2])]
            sraf_width_array = sraf_width_array.reshape(-1)

        sraf_width_array = [float(w) for w in sraf_width_array]

        full_sraf_mask = full_sraf_mask.astype(bool)
        H, W = full_sraf_mask.shape
        final_coverage = np.zeros((H, W), dtype=np.float32)
        #产生子像素采样点
        sample_offsets = generate_sample_offsets(samples)

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
                coverage = rasterize_distance_msaa_single(
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
            
            #一阶一阶处理
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

                    coverage = rasterize_distance_msaa_single(
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


# 基于参数化曲线的SRAF渲染


def distance_msaa_multi_curves(
    fitted_curves_by_order,  # 输入：{order_idx: [array(200, 2), ...]}
    sraf_width_array,        # 输入：对应的宽度数组
    mask_shape,              # 画布尺寸 (H, W)
    samples=4,               # MSAA 采样数
    show=False
):
    """
    主入口：输入每一阶拟合后的 B 样条曲线坐标点，输出合并后的 MSAA 渲染图。
    """
    # 1. 参数格式化
    sraf_width_array = np.asarray(sraf_width_array).flatten()
    # 使用你原有的生成采样偏移的函数
    sample_offsets = generate_sample_offsets(samples) 
    
    H, W = mask_shape
    final_coverage = np.zeros((H, W), dtype=np.float32)

    # 2. 遍历每一阶（Order）进行渲染
    # 假设 sraf_width_array 的索引与 order 的顺序对应（或 order 是从 1 开始的整数）
    for i, (order, curves_list) in enumerate(fitted_curves_by_order.items()):
        # 获取当前阶的半宽
        # 如果 width_array 只有 1 个值，则共用；否则按索引取
        w = sraf_width_array[0] if len(sraf_width_array) == 1 else sraf_width_array[i]
        hw = float(w) 
        
        if not curves_list:
            continue

        # 3. 渲染该阶下的每一根独立 SRAF 曲线
        for curve_pts in curves_list:
            # 这里的 curve_pts 是你拟合出的 (200, 2) 浮点坐标数组
            coverage = rasterize_distance_msaa_single_from_pts(
                curve_pts,
                hw,
                sample_offsets,
                mask_shape
            )
            
            # 合并结果（取最大值以处理不同线段、不同阶之间的重叠）
            final_coverage = np.maximum(final_coverage, coverage)

    # 4. 可视化
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(final_coverage, cmap='gray'); plt.title("MSAA Curve Rendering")
        plt.subplot(1, 2, 2); plt.imshow(final_coverage > 0.5, cmap='gray'); plt.title("Binary Result")
        plt.show()

    return final_coverage

def rasterize_distance_msaa_single_from_pts(
        curve_pts,      # (200, 2) array
        half_width, 
        sample_offsets, 
        mask_shape
    ):
    H, W = mask_shape
    
    # --- 1. 局部化处理：计算 Bounding Box 减少计算量 ---
    # 增加 padding 确保距离场覆盖完整宽度
    pad = int(half_width + 2)
    y_min = max(0, int(np.floor(curve_pts[:, 0].min() - pad)))
    y_max = min(H, int(np.ceil(curve_pts[:, 0].max() + pad)))
    x_min = max(0, int(np.floor(curve_pts[:, 1].min() - pad)))
    x_max = min(W, int(np.ceil(curve_pts[:, 1].max() + pad)))
    
    if y_max <= y_min or x_max <= x_min:
        return np.zeros(mask_shape, dtype=np.float32)

    # --- 2. 建立局部距离场 ---
    # 生成局部区域像素中心坐标
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    pixel_centers = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
    # 使用 KDTree 计算精确的欧氏距离
    tree = cKDTree(curve_pts)
    d_local, _ = tree.query(pixel_centers, k=1)
    dist_map_local = d_local.reshape(yy.shape)

    # --- 3. 窄带逻辑 (完全保留你之前的优化策略) ---
    r_pixel = np.sqrt(2) / 2
    inside = dist_map_local <= (half_width - r_pixel)
    outside = dist_map_local >= (half_width + r_pixel)
    band = ~(inside | outside)

    # 初始化局部覆盖率图
    coverage_local = np.zeros(yy.shape, dtype=np.float32)
    coverage_local[inside] = 1.0

    # --- 4. 窄带内的子像素 MSAA ---
    rows_in_band, cols_in_band = np.nonzero(band)
    if len(rows_in_band) > 0:
        S = sample_offsets.shape[0]
        # 计算窄带内所有子像素的全局坐标
        # (rows_in_band + y_min) 转回全局 y 坐标
        samples_y = (rows_in_band + y_min)[:, None] + sample_offsets[:, 0]
        samples_x = (cols_in_band + x_min)[:, None] + sample_offsets[:, 1]
        sub_samples = np.stack([samples_y, samples_x], axis=-1).reshape(-1, 2)

        # 直接通过 KDTree 查询子像素是否在宽度内
        d_sub, _ = tree.query(sub_samples, k=1)
        inside_sub = (d_sub <= half_width).astype(np.float32)

        # 计算窄带像素的平均覆盖率
        inside_sub = inside_sub.reshape(len(rows_in_band), S)
        coverage_local[rows_in_band, cols_in_band] = inside_sub.mean(axis=1)

    # --- 5. 写回全图画布 ---
    full_coverage = np.zeros(mask_shape, dtype=np.float32)
    full_coverage[y_min:y_max, x_min:x_max] = coverage_local

    return full_coverage



def rasterize_skeleton_msaa_single(
        skeleton,            # 输入：单阶或单根 SRAF 的骨架线 (Boolean Array)
        half_width,         # 输入：该阶 SRAF 的半宽
        sample_offsets, 
        batch_size=None,
        max_mem_fraction=0.2
    ):
    """
    通过骨架线和目标半宽，利用距离场和 MSAA 渲染出带灰度的 SRAF。
    """
    H, W = skeleton.shape

    # --- 1. 直接由骨架线计算距离场 ---
    # 骨架线上点到骨架线的距离为 0，背景点计算到骨架线的最短欧氏距离
    dist_map = distance_transform_edt(~skeleton)

    # --- 2. 确定需要进行 MSAA 处理的窄带区域 ---
    # r_pixel 定义了采样范围
    r_pixel = np.sqrt(2) / 2
    inside = dist_map <= (half_width - r_pixel)
    outside = dist_map >= (half_width + r_pixel)
    band = ~(inside | outside) # 只有边界窄带需要进行子像素采样
    # plt.figure(figsize=(6, 6))
    # plt.imshow(skeleton, cmap='gray')
    # plt.imshow(band, cmap='hot', alpha=0.7)  # 窄带叠加
    # plt.axis('off')
    # plt.title('Skeleton with MSAA Band')
    # plt.show()
    
    # 初始化覆盖率矩阵
    coverage = np.zeros((H, W), dtype=np.float32)
    coverage[inside] = 1.0  # 内部完全覆盖

    # 找到窄带中的像素坐标
    rows, cols = np.nonzero(band)
    if len(rows) == 0:
        return coverage

    # --- 3. 子像素采样 (MSAA) ---
    S = sample_offsets.shape[0]
    # 将窄带像素坐标偏移到子像素位置
    samples_y = rows[:, None] + sample_offsets[:, 0]
    samples_x = cols[:, None] + sample_offsets[:, 1]
    samples = np.stack([samples_y, samples_x], axis=-1).reshape(-1, 2)

    # 判断子像素点是否在 half_width 范围内
    # 注意：这里调用你现有的 vectorized_is_within_width 函数
    inside_sub = vectorized_is_within_width(
        dist_map,
        samples,
        half_width,
        batch_size=batch_size,
        max_mem_fraction=max_mem_fraction
    )

    # 计算每个像素内子像素的平均值，得到灰度覆盖率
    inside_sub = inside_sub.reshape(len(rows), S)
    coverage[rows, cols] = inside_sub.mean(axis=1)

    return coverage
def distance_msaa_multi_skeleton(
        skeleton_layers,    # 输入：字典 {order_idx: skeleton_mask}
        sraf_width_array,   # 输入：对应的宽度数组
        samples=4,
        show=False
    ):
    """
    主入口：输入每一阶的骨架线，输出合并后的渲染图
    根据骨架图进行sraf宽度的初始化，采用msaa的方式进行轮廓处理
    """
    # 1. 参数格式化
    sraf_width_array = np.asarray(sraf_width_array).flatten()
    sample_offsets = generate_sample_offsets(samples)
    
    # 预准备：获取画布尺寸（假设所有骨架图尺寸一致）
    first_key = list(skeleton_layers.keys())[0]
    H, W = skeleton_layers[first_key].shape
    final_coverage = np.zeros((H, W), dtype=np.float32)

    # 2. 遍历每一阶骨架线进行渲染
    # 假设 sraf_width_array 的索引与 skeleton_layers 的 key 对应
    for i, (order, skel_mask) in enumerate(skeleton_layers.items()):
        # 获取当前阶的半宽
        # 注意：如果 sraf_width_array 长度为 1，则所有阶共用一个宽度
        hw = float(sraf_width_array[0]) if len(sraf_width_array) == 1 else float(sraf_width_array[i])
        
        if skel_mask.sum() == 0:
            continue

        # 3. 对当前阶内的连通域进行分割（防止不同线段距离场互相干扰，虽然通常骨架线不需要，但为了严谨保留）
        labeled, num = skm.label(skel_mask.astype(bool), connectivity=2, return_num=True)

        for idx in range(1, num + 1):
            single_skel = (labeled == idx)
            
            # 渲染单根骨架线
            coverage = rasterize_skeleton_msaa_single(
                single_skel,
                hw,
                sample_offsets
            )
            
            # 合并结果（取最大值以处理重叠部分）
            final_coverage = np.maximum(final_coverage, coverage)

    # 4. 可视化
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(final_coverage, cmap='gray'); plt.title("MSAA Rendering")
        plt.subplot(1, 2, 2); plt.imshow(final_coverage > 0.5, cmap='gray'); plt.title("Binary Result")
        plt.show()

    return final_coverage  
def _save_with_colorbar(image, save_path, cmap, dpi: int = 300, title: str = None,
                        pixel_size=None):
        """用 imshow + colorbar 保存一张二维数组的可视化图，自带颜色条。

        - pixel_size 为 None: 关闭坐标轴 (旧风格);
        - pixel_size 给定: 坐标轴以 nm 为单位、以矩阵中心为原点
          (coord(i) = (i - (N-1)/2) * pixel_size)。
        - 输出 dpi 由参数控制；保存后立即关闭 figure 防内存累积。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.asarray(image)
        fig, ax = plt.subplots(figsize=(5, 5))

        if pixel_size is not None:
            h, w = arr.shape[:2]
            ps = float(pixel_size)
            x_half = (w - 1) / 2.0 * ps
            y_half = (h - 1) / 2.0 * ps
            # origin='upper' (默认), y extent 写成 [top, bottom] 使 y 轴上正下负且不翻转
            extent = [-x_half - ps / 2.0, x_half + ps / 2.0,
                      y_half + ps / 2.0, -y_half - ps / 2.0]
            im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
        else:
            im = ax.imshow(arr, cmap=cmap)
            ax.set_axis_off()

        if title:
            ax.set_title(str(title))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_width_results(
        file_path,current_mask,current_SRAFs,aI,wI,width, mask_color,
        pixel_size=None):
        base_dir = Path(file_path)

        # ===== 统一 & 合理的目录结构 =====
        sub_dirs = ['current_mask','current_mask_txt','current_SRAFs', 'current_SRAFs_txt','ai','ai_txt','wi','wi_txt','width','error',]

        # 创建目录
        for sub in sub_dirs:
            (base_dir / sub).mkdir(parents=True, exist_ok=True)

        # ===== 文件路径 =====
        mask_png_path   = base_dir / 'current_mask' / 'latest.png'
        sraf_png_path   = base_dir / 'current_SRAFs' / 'latest.png'
        ai_png_path     = base_dir / 'ai' / 'latest.png'
        wi_png_path     = base_dir / 'wi' / 'latest.png'
        
        mask_txt_path   = base_dir / 'current_mask_txt' / 'latest.txt'
        sraf_txt_path   = base_dir / 'current_SRAFs_txt' / 'latest.txt'
        ai_txt_path     = base_dir / 'ai_txt' / 'latest.txt'
        wi_txt_path     = base_dir / 'wi_txt' / 'latest.txt'
        width_txt_path  = base_dir / 'width' / 'latest.txt'
        

        # ===== 保存 txt =====
       # ===== 保存 txt（自动覆盖同名文件） =====
        np.savetxt(mask_txt_path, current_mask, fmt='%.6f')
        np.savetxt(sraf_txt_path, current_SRAFs, fmt='%.6f')
        np.savetxt(ai_txt_path, aI, fmt='%.6f')
        np.savetxt(wi_txt_path, wI, fmt='%.6f')
        np.savetxt(width_txt_path, width.reshape(-1, 1), fmt='%.3f')
        
        # with open(f'{cps_txt_path}/第{it}次的控制点坐标.txt', 'w') as f:
        #     for row in current_cps:
        #         f.write(' '.join(map(str, row)) + '\n')


        # ===== 保存 png（给定 pixel_size 则带 nm 坐标轴）=====
        _save_with_colorbar(current_mask,  mask_png_path, cmap=mask_color, dpi=600, pixel_size=pixel_size)
        _save_with_colorbar(current_SRAFs, sraf_png_path, cmap=mask_color, dpi=600, pixel_size=pixel_size)
        _save_with_colorbar(aI,            ai_png_path,   cmap=mask_color, dpi=600, pixel_size=pixel_size)
        _save_with_colorbar(wI,            wi_png_path,   cmap=mask_color, dpi=600, pixel_size=pixel_size)

        # for order, mask in sraf_layer.items():
        #     self._save_with_colorbar(mask, sraf_png_path, cmap=mask_color)




import re
def read_cps_txt(path):
    """读取控制点 txt，返回 ragged list: [ndarray(N_i, 2), ...]。

    注意：不同轮廓点数通常不一致，不能直接外层 np.array(curves)，
    否则在新版本 numpy 中会触发 inhomogeneous shape ValueError。
    """
    curves = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 匹配所有 [x y]
            matches = re.findall(r'\[([^\]]+)\]', line)
            if not matches:
                continue

            curve = []
            for m in matches:
                x, y = map(float, m.split())
                curve.append([x, y])

            curves.append(np.asarray(curve, dtype=np.float64))

    return curves