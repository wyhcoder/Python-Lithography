# from nt import times_result
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import threading
from tqdm import tqdm
import time
from pathlib import Path
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v1, images_simulation_v2, images_simulation_v3
from litho_model.load_mask_bmp import load_mask_image
from op_model.demo_epe_wepe import caculate_epe, caculate_wepe
from utils_model.demo_MSAA import AntiAliasRenderer
from utils_model.demo_parametric import ParametricDemo
from utils_model.demo_meef_utils import save_iteration_results,save_excel,drawcostcurve,show_sample_mcps, get_new_cps,get_cp_vectors,highlight_contour_red, _save_images,move_single_control_point,find_turelambadas,truncated_svd_solver,move_single_control_point_X,move_single_control_point_Y,get_new_cps_xy,save_matrix_nm,save_bspline_curve_mask
from concurrent.futures import ThreadPoolExecutor, as_completed


def _save_bspline_curves(parametric, new_cps, iteration_idx, filepath):
    """
    将每次迭代的 B 样条曲线 (渲染为像素 mask 之前) 保存到 txt 文件。

    Parameters
    ----------
    parametric : ParametricDemo
    new_cps : list of ndarray  当前迭代的控制点列表
    iteration_idx : int  迭代编号
    filepath : str or Path  MEEF 输出根目录
    """
    curve_dir = Path(filepath) / "curves" / "bspline_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    try:
        curves = parametric.get_curve_points(new_cps)
        out_path = curve_dir / f"bspline_iter_{iteration_idx:03d}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# B-spline curve at iteration {iteration_idx}\n")
            f.write(f"# curve_type = {parametric.curve_type}\n")
            f.write(f"# num_contours = {len(curves)}\n")
            for ci, cv in enumerate(curves):
                f.write(f"# contour {ci}, num_points = {len(cv)}\n")
                np.savetxt(f, cv, fmt="%.6f", delimiter=" ")
    except Exception as e:
        print(f"  [warn] 保存 B 样条曲线失败 (iter {iteration_idx}): {e}")


def _save_cp_history(new_cps, iteration_idx, filepath):
    """
    将每次迭代的 B 样条控制点 (CP) 原始坐标保存到 txt 文件,
    供后续绘制 CP 移动轨迹使用。

    与 _save_bspline_curves 不同: 这里保存的是 CP 本身 (数量少),
    而不是渲染后的采样曲线点。

    存放路径: <filepath>/curves/cp_history/cp_iter_XXX.txt
    格式与 bspline_iter_XXX.txt 一致, 可复用 _load_bspline_curves_from_txt 解析。

    Parameters
    ----------
    new_cps : list of ndarray  每个轮廓的 CP 坐标, 形状 (K_i, 2)
    iteration_idx : int        迭代编号 (0=初始)
    filepath : str or Path     MEEF 输出根目录
    """
    cp_dir = Path(filepath) / "curves" / "cp_history"
    cp_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_path = cp_dir / f"cp_iter_{iteration_idx:03d}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# CP positions at iteration {iteration_idx}\n")
            f.write(f"# num_contours = {len(new_cps)}\n")
            for ci, cps in enumerate(new_cps):
                cps_arr = np.asarray(cps)
                if cps_arr.ndim == 1:
                    cps_arr = cps_arr.reshape(-1, 2)
                f.write(f"# contour {ci}, num_points = {len(cps_arr)}\n")
                np.savetxt(f, cps_arr, fmt="%.6f", delimiter=" ")
    except Exception as e:
        print(f"  [warn] 保存 CP 历史失败 (iter {iteration_idx}): {e}")


def _load_bspline_curves_from_txt(txt_path):
    """从 bspline_iter_XXX.txt 加载曲线列表 (list of (N_i, 2) ndarray)。"""
    curves = []
    current = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# contour"):
                if current:
                    curves.append(np.array(current, dtype=np.float64))
                    current = []
                continue
            if line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) >= 2:
                current.append([float(vals[0]), float(vals[1])])
    if current:
        curves.append(np.array(current, dtype=np.float64))
    return curves


def _plot_bspline_evolution(filepath, target_mask, sample_iters=None):
    """
    把多次迭代的 B 样条曲线叠在 target mask 上, 画演化对比图。

    Parameters
    ----------
    filepath : str or Path
        MEEF 输出根目录 (内含 bspline_curves/ 子目录)
    target_mask : np.ndarray  (H, W)
        目标版图 (作为底图)
    sample_iters : list[int] or None
        要画出的迭代号; None 表示自动采样 (起点 + 若干中间 + 终点)
    """
    curve_dir = Path(filepath) / "curves" / "bspline_curves"
    if not curve_dir.exists():
        return

    files = sorted(curve_dir.glob("bspline_iter_*.txt"))
    if not files:
        return

    iters_available = []
    for fp in files:
        try:
            it = int(fp.stem.split("_")[-1])
            iters_available.append((it, fp))
        except ValueError:
            continue
    iters_available.sort(key=lambda x: x[0])

    # 自动采样: 等间隔取 ~6 个
    if sample_iters is None:
        n = len(iters_available)
        if n <= 6:
            picked = iters_available
        else:
            idxs = np.linspace(0, n - 1, 6).round().astype(int)
            picked = [iters_available[i] for i in idxs]
    else:
        wanted = set(sample_iters)
        picked = [(it, fp) for it, fp in iters_available if it in wanted]

    if not picked:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(target_mask, cmap="gray", origin="upper", alpha=0.45)

    cmap = plt.cm.get_cmap("viridis", len(picked))
    for k, (it, fp) in enumerate(picked):
        try:
            curves = _load_bspline_curves_from_txt(fp)
        except Exception as e:
            print(f"  [warn] 加载 {fp.name} 失败: {e}")
            continue
        color = cmap(k)
        label = f"iter {it}"
        for ci, cv in enumerate(curves):
            # cv shape (N, 2) [y, x]; matplotlib plot 用 (x, y)
            xs = cv[:, 1]; ys = cv[:, 0]
            # 闭合: 把首点追加到尾
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
            ax.plot(xs, ys, color=color, linewidth=1.4,
                    label=label if ci == 0 else None)

    ax.set_title("B-spline curve evolution (overlaid on target)")
    ax.set_aspect("equal")
    ax.set_xlim(0, target_mask.shape[1])
    ax.set_ylim(target_mask.shape[0], 0)  # y 向下与 imshow 一致
    ax.legend(loc="upper right", fontsize=9)
    out_dir = Path(filepath) / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bspline_evolution.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] B 样条曲线演化图: {out_path}")


# _litho_lock = threading.Lock()
# def _build_meef_matrix_parallel_threads(simulator, 
#                                         parametric,
#                                         current_vectors, 
#                                         current_cps, 
#                                         target_mask, 
#                                         sraf_mask, 
#                                         eps, 
#                                         delta, 
#                                         render,
#                                         curve_type):
#     num_eps = len(eps)
#     num_cps = sum(len(v) for v in current_vectors)
#     M = np.zeros((num_eps, num_cps), dtype=float)
#     cp_offsets = []
#     _off = 0
#     for v in current_vectors:
#         cp_offsets.append(_off)
#         _off += len(v)
#     def _worker(contour_idx, cp_idx, vector, old_cp):
#             # 每个任务独立构造局部 cps，不影响全局
#             local_cps = [contour.copy() for contour in current_cps]
#             opt_type = curve_type
#             if opt_type == 'OA':
#                 #外扰动
#                 new_cp_out = move_single_control_point(old_cp, delta, vector.vector)
#                 local_cps[contour_idx][cp_idx] = new_cp_out
#                 mask_out = render.MSAA(local_cps,target_mask, 'gray') + sraf_mask
#                 #内扰动
#                 new_cp_in = move_single_control_point(old_cp, -delta, vector.vector)
#                 mask_in = render.MSAA(local_cps,target_mask, 'gray') + sraf_mask
            
#             elif opt_type == 'BS':  
#                 # 外扰动
#                 new_cp_out = move_single_control_point(old_cp, delta, vector.vector)
#                 local_cps[contour_idx][cp_idx] = new_cp_out
#                 BS_contour = parametric.b_spile(local_cps, 0.3, num_points=200)
#                 mask_out = render.MSAA(BS_contour, target_mask, 'gray') + sraf_mask
#                 #内扰动
#                 local_cps[contour_idx][cp_idx] = move_single_control_point(old_cp, -delta, vector.vector)
#                 BS_contour = parametric.b_spile(local_cps, 0.3, num_points=200)
#                 mask_in = render.MSAA(BS_contour, target_mask, 'gray') + sraf_mask
                
#             elif opt_type == 'BZ':
#                 # 外扰动
#                 new_cp_out = move_single_control_point(old_cp, delta, vector.vector)
#                 local_cps[contour_idx][cp_idx] = new_cp_out
#                 BZ_contour = parametric.compute_BZCPandBZ_points(local_cps, target_mask)
#                 mask_out = render.MSAA(BS_contour, target_mask, 'gray') + sraf_mask
#                 #内扰动
#                 local_cps[contour_idx][cp_idx] = move_single_control_point(old_cp, -delta, vector.vector)
#                 BZ_contour = parametric.compute_BZCPandBZ_points(local_cps, target_mask)
#                 mask_in = render.MSAA(BZ_contour, target_mask, 'gray') + sraf_mask
#             def _epe_from_mask(mask):
#                 with _litho_lock:
#                     ai_image, wafer_image, intermediates = images_simulation_v2(
#                     mask_spatial=mask,
#                     resist_params=simulator.params.resist,
#                     opt_cache=simulator.opt_cache
#                     )
#                     _,epe_vector = caculate_epe(ai_image, simulator.params.resist.threshold, eps,num_eps, None, False)
#                 return epe_vector   
#             epe_out = _epe_from_mask(mask_out)
#             epe_in  = _epe_from_mask(mask_in)
#             # 中心差分
#             n = (epe_out - epe_in) / (2  * 4 * delta)
#             n = np.round(n, 6)  # 保持你原来的行为
#             col = cp_offsets[contour_idx] + cp_idx
#             return col, n.reshape(-1)
#     max_workers = min(7, max(1, os.cpu_count() or 8))
#     futures = []
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#             for contour_idx, vectors in enumerate(current_vectors):
#                 for cp_idx, vector in enumerate(vectors):
#                     old_cp = current_cps[contour_idx][cp_idx]
#                     futures.append(ex.submit(_worker, contour_idx, cp_idx, vector, old_cp))

#             for fut in tqdm(as_completed(futures), total=len(futures),
#                             desc="Building MEEF matrix", ncols=100, ascii=" █"):
#                 col, n = fut.result()
#                 M[:, col] = n

#     return M
# _litho_lock = threading.Lock()

# def _meef_task_worker(args):
#     """
#     独立的 Worker 函数，负责计算单个控制点对 MEEF 矩阵列的贡献。
#     """
#     (contour_idx, cp_idx, col_idx, vector_obj, old_cp, current_cps, 
#      target_mask, sraf_mask, eps, delta, render, parametric, 
#      curve_type, simulator_params, opt_cache,pixel_size) = args

#     # 导入必要的函数（如果在类外定义，确保能访问到这些工具函数）
#     # from utils_model.demo_meef_utils import move_single_control_point
#     # from litho_model.demo_compute_image import images_simulation_v2
#     # from op_model.demo_epe_wepe import caculate_epe

#     num_eps = len(eps)

#     def _get_epe_vector(perturbation):
#         # 1. 移动控制点
#         local_cps = [contour.copy() for contour in current_cps]
#         new_cp = move_single_control_point(old_cp, perturbation, vector_obj.vector)
#         local_cps[contour_idx][cp_idx] = new_cp

#         # 2. 根据曲线类型生成渲染轮廓
#         if curve_type == 'BS':
#             render_contour = parametric.b_spile(local_cps, 0.3, num_points=200)
#         elif curve_type == 'BZ':
#             # 修正了原代码中 BZ 模式下引用 BS_contour 的错误
#             render_contour = parametric.compute_BZCPandBZ_points(local_cps, target_mask)
#         else: # OA (Original/Anchor)
#             render_contour = local_cps

#         # 3. 渲染 Mask
#         mask = render.MSAA(render_contour, target_mask, 'gray') + sraf_mask

#         # 4. 运行仿真 (加锁确保 opt_cache 安全)
#         with _litho_lock:
#             ai_image, _, _ = images_simulation_v2(
#                 mask_spatial=mask,
#                 resist_params=simulator_params,
#                 opt_cache=opt_cache
#             )
#             # 获取 EPE 向量 (caculate_epe 返回: mean_epe, epe_vector)
#             _, epe_vector = caculate_epe(
#                 ai_image, 
#                 simulator_params.threshold, 
#                 eps, 
#                 num_eps, 
#                 None, 
#                 False
#             )
#         return epe_vector

#     # 执行中心差分计算梯度
#     epe_out = _get_epe_vector(delta)
#     epe_in  = _get_epe_vector(-delta)
    
#     # 计算列梯度: (EPE+ - EPE-) / (2 * delta)
#     # 注意：根据你的物理模型需求，确认分母是否需要额外的 4 倍系数
#     # column_gradient = (epe_out - epe_in) / (2 *  pixel_size * delta)
#     column_gradient = (epe_out - epe_in) / (2  * delta)
#     column_gradient = np.round(column_gradient, 6)

#     return col_idx, column_gradient.flatten()

# def _build_meef_matrix_parallel_threads(simulator, 
#                                         parametric,
#                                         current_vectors, 
#                                         current_cps, 
#                                         target_mask, 
#                                         sraf_mask, 
#                                         eps, 
#                                         delta, 
#                                         render,
#                                         curve_type,
#                                         pixel_size):
#     """
#     改进后的并行构建 MEEF 矩阵函数
#     """
#     num_eps = len(eps)
#     num_cps = sum(len(v) for v in current_vectors)
#     M = np.zeros((num_eps, num_cps), dtype=float)
    
#     # 准备任务参数列表
#     tasks = []
#     global_col_counter = 0
    
#     # 预先提取常用参数减少 worker 内部访问开销
#     sim_params = simulator.params.resist
#     opt_cache = simulator.opt_cache
    

#     for contour_idx, vectors in enumerate(current_vectors):
#         for cp_idx, vector in enumerate(vectors):
#             old_cp = current_cps[contour_idx][cp_idx]
            
#             # 打包所有必要数据
#             task_args = (
#                 contour_idx, 
#                 cp_idx, 
#                 global_col_counter,
#                 vector, 
#                 old_cp, 
#                 current_cps, 
#                 target_mask, 
#                 sraf_mask, 
#                 eps, 
#                 delta, 
#                 render, 
#                 parametric, 
#                 curve_type, 
#                 sim_params, 
#                 opt_cache,
#                 pixel_size
#             )
#             tasks.append(task_args)
#             global_col_counter += 1

#     # 设置并行线程数 (建议根据 CPU 核心数和仿真负载调整)
#     max_workers = min(7, os.cpu_count() or 4)
    
#     # print(f"Starting MEEF matrix build with {max_workers} threads...")
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_col = {executor.submit(_meef_task_worker, arg): arg[2] for arg in tasks}
        
#         for future in tqdm(as_completed(future_to_col), total=len(tasks),
#                           desc="Building MEEF matrix", ncols=100, ascii=" █"):
#             try:
#                 col_idx, grad_vector = future.result()
#                 # 填充矩阵对应的列
#                 M[:, col_idx] = grad_vector
#             except Exception as e:
#                 print(f"\nError computing MEEF for column {future_to_col[future]}: {e}")

#     return M


def _meef_perturbation_workerV2(args):
    """
    X/Y 解耦版 MEEF 扰动 worker。

    与角平分线版 `_meef_perturbation_worker` 的唯一区别是: 扰动方向不再
    沿控制点法向(角平分线), 而是严格沿 X 轴或 Y 轴, 由 `move_type` 指定
    ('X_direction' / 'Y_direction')。返回值额外带回 `move_type`, 以便
    上层把结果分别累加到 Mx / My 两个矩阵中。
    """
    (col_idx, side, contour_idx, cp_idx, old_cp, current_cps,
    target_mask, sraf_mask, eps, delta, render, parametric,
    curve_type, move_type, simulator_params, opt_cache) = args

    num_eps = len(eps)
    perturbation = delta if side == +1 else -delta

    local_cps = [contour.copy() for contour in current_cps]
    if move_type == 'X_direction':
        new_cp = move_single_control_point_X(old_cp, perturbation)
    else:
        new_cp = move_single_control_point_Y(old_cp, perturbation)
    local_cps[contour_idx][cp_idx] = new_cp

    if curve_type == 'BS':
        render_contour = parametric.b_spile(local_cps, 0.3, num_points=200)
    elif curve_type == 'BZ':
        render_contour = parametric.compute_BZCPandBZ_points(local_cps, target_mask)
    else:
        render_contour = local_cps

    # 将轮廓光栅化为 mask，并叠加 SRAF
    mask = render.MSAA(render_contour, target_mask, 'gray') + sraf_mask
    # 运行前向仿真，得到 aerial image (线程安全版)
    ai_image, _, _ = images_simulation_v3(
        mask_spatial=mask,
        resist_params=simulator_params,
        opt_cache=opt_cache,
    )
    # 从 aerial image 中提取当前扰动对应的 EPE 向量
    _, epe_vector = caculate_epe(
        ai_image,
        simulator_params.threshold,
        eps,
        num_eps,
        None,
        False,
    )
    return col_idx, side, move_type, epe_vector.flatten()


def _build_meef_matrix_xy_parallel_threads(simulator,
                                           parametric,
                                           current_cps,
                                           target_mask,
                                           sraf_mask,
                                           eps,
                                           delta,
                                           render,
                                           curve_type):
    """
    X/Y 解耦地并行构建两套 MEEF 矩阵 Mx 与 My。

    对每个控制点分别做 4 个独立任务:
        (X, +delta), (X, -delta), (Y, +delta), (Y, -delta)
    分别用中心差分得到:
        Mx[:, col] = (epe_x+ - epe_x-) / (2*delta)
        My[:, col] = (epe_y+ - epe_y-) / (2*delta)

    Mx / My 形状均为 (num_eps, num_cps)。后续分别求解可得到每个控制点的
    Δx 与 Δy, 合成即为真实的二维移动向量(方向 + 步长)。
    """
    num_eps = len(eps)
    num_cps = sum(len(c) for c in current_cps)

    Mx = np.zeros((num_eps, num_cps), dtype=float)
    My = np.zeros((num_eps, num_cps), dtype=float)

    # 缓冲: 每个方向两侧各存一份 EPE 向量
    epe_x_plus = np.zeros((num_cps, num_eps), dtype=float)
    epe_x_minus = np.zeros((num_cps, num_eps), dtype=float)
    epe_y_plus = np.zeros((num_cps, num_eps), dtype=float)
    epe_y_minus = np.zeros((num_cps, num_eps), dtype=float)

    sim_params = simulator.params.resist
    opt_cache = simulator.opt_cache

    # 收集 (列, 方向, 侧) 的细粒度任务
    tasks = []
    global_col_counter = 0
    for contour_idx, contour in enumerate(current_cps):
        for cp_idx in range(len(contour)):
            old_cp = current_cps[contour_idx][cp_idx]
            for move_type in ('X_direction', 'Y_direction'):
                for side in (+1, -1):
                    tasks.append((
                        global_col_counter,
                        side,
                        contour_idx,
                        cp_idx,
                        old_cp,
                        current_cps,
                        target_mask,
                        sraf_mask,
                        eps,
                        delta,
                        render,
                        parametric,
                        curve_type,
                        move_type,
                        sim_params,
                        opt_cache,
                    ))
            global_col_counter += 1

    if not tasks:
        return Mx, My

    max_workers = min(len(tasks), max(1, os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_meef_perturbation_workerV2, arg): (arg[0], arg[14], arg[1])
            for arg in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks),
                           desc="Building MEEF matrix (X/Y)", ncols=100, ascii=" █"):
            try:
                col_idx, side, move_type, epe_vec = future.result()
                if move_type == 'X_direction':
                    if side == +1:
                        epe_x_plus[col_idx] = epe_vec
                    else:
                        epe_x_minus[col_idx] = epe_vec
                else:
                    if side == +1:
                        epe_y_plus[col_idx] = epe_vec
                    else:
                        epe_y_minus[col_idx] = epe_vec
            except Exception as e:
                fail_col, fail_move, fail_side = future_to_task[future]
                print(f"\nError computing MEEF(X/Y) col {fail_col} "
                      f"{fail_move} side {fail_side}: {e}")

    # 中心差分聚合
    grad_x = np.round((epe_x_plus - epe_x_minus) / (2.0 * delta), 6)
    grad_y = np.round((epe_y_plus - epe_y_minus) / (2.0 * delta), 6)
    Mx[:] = grad_x.T
    My[:] = grad_y.T

    return Mx, My

# def _build_meef_matrix_parallel_threads_V2(simulator,
#                                         parametric,
#                                         current_vectors,
#                                         current_cps,
#                                         target_mask,
#                                         sraf_mask,
#                                         eps,
#                                         delta,
#                                         render,
#                                         curve_type,
#                                         pixel_size):
#     num_eps = len(eps)
#     num_cps = sum(len(v) for v in current_vectors)

#     # M 的形状为: [EPE采样点数, 控制点总数]
#     M = np.zeros((num_eps, num_cps), dtype=float)

#     # epe 缓冲区: 每列两侧 (+delta, -delta) 各存一个 EPE 向量
#     epe_plus = np.zeros((num_cps, num_eps), dtype=float)
#     epe_minus = np.zeros((num_cps, num_eps), dtype=float)

#     # 提前取出常用参数，减少 worker 内部的属性访问开销
#     sim_params = simulator.params.resist
#     opt_cache = simulator.opt_cache




    


def _meef_perturbation_worker(args):
    """
    并行构建 MEEF 矩阵时的最小粒度 worker。

    每个任务只负责计算"某一控制点 + 某一侧扰动(+delta 或 -delta)"对应的
    EPE 向量。这样把原来"一列两次仿真"拆成两个独立任务，调度粒度翻倍，
    多核 CPU 利用率更高。
    """
    (col_idx, side, contour_idx, cp_idx, vector_obj, old_cp, current_cps,
     target_mask, sraf_mask, eps, delta, render, parametric,
     curve_type, simulator_params, opt_cache) = args

    num_eps = len(eps)
    perturbation = delta if side == +1 else -delta

    # 为当前线程复制一份控制点，避免修改共享数据
    local_cps = [contour.copy() for contour in current_cps]

    # 沿当前控制点的法向/优化方向做一次微扰
    new_cp = move_single_control_point(old_cp, perturbation, vector_obj.vector)
    local_cps[contour_idx][cp_idx] = new_cp

    # 根据曲线类型生成用于光刻仿真的轮廓
    if curve_type == 'BS':
        render_contour = parametric.b_spile(local_cps, 0.3, num_points=200)
    elif curve_type == 'BZ':
        render_contour = parametric.compute_BZCPandBZ_points(local_cps, target_mask)
    else:
        render_contour = local_cps

    # 将轮廓光栅化为 mask，并叠加 SRAF
    mask = render.MSAA(render_contour, target_mask, 'gray') + sraf_mask

    # 运行前向仿真，得到 aerial image (线程安全版)
    ai_image, _, _ = images_simulation_v3(
        mask_spatial=mask,
        resist_params=simulator_params,
        opt_cache=opt_cache,
    )

    # 从 aerial image 中提取当前扰动对应的 EPE 向量
    _, epe_vector = caculate_epe(
        ai_image,
        simulator_params.threshold,
        eps,
        num_eps,
        None,
        False,
    )
    return col_idx, side, epe_vector.flatten()


def _build_meef_matrix_parallel_threads(simulator,
                                        parametric,
                                        current_vectors,
                                        current_cps,
                                        target_mask,
                                        sraf_mask,
                                        eps,
                                        delta,
                                        render,
                                        curve_type,
                                        pixel_size):
    """
    使用线程池并行构建 MEEF 矩阵。

    粒度策略：把每个控制点的 +delta / -delta 两次扰动拆成两个独立任务，
    任务总数 = 2 * num_cps。前向仿真依赖线程安全版 images_simulation_v3，
    因此多线程可以真正并发执行。
    """
    num_eps = len(eps)
    num_cps = sum(len(v) for v in current_vectors)

    # M 的形状为: [EPE采样点数, 控制点总数]
    M = np.zeros((num_eps, num_cps), dtype=float)

    # epe 缓冲区: 每列两侧 (+delta, -delta) 各存一个 EPE 向量
    epe_plus = np.zeros((num_cps, num_eps), dtype=float)
    epe_minus = np.zeros((num_cps, num_eps), dtype=float)

    # 提前取出常用参数，减少 worker 内部的属性访问开销
    sim_params = simulator.params.resist
    opt_cache = simulator.opt_cache

    # 收集所有(列, 侧)的细粒度任务
    tasks = []
    global_col_counter = 0
    for contour_idx, vectors in enumerate(current_vectors):
        for cp_idx, vector in enumerate(vectors):
            old_cp = current_cps[contour_idx][cp_idx]

            # 同一列拆出 +delta 和 -delta 两个独立任务
            for side in (+1, -1):
                tasks.append((
                    global_col_counter,
                    side,
                    contour_idx,
                    cp_idx,
                    vector,
                    old_cp,
                    current_cps,
                    target_mask,
                    sraf_mask,
                    eps,
                    delta,
                    render,
                    parametric,
                    curve_type,
                    sim_params,
                    opt_cache,
                ))
            global_col_counter += 1

    # 没有任务时直接返回空矩阵
    if not tasks:
        return M

    # 线程数不超过任务数，也不超过机器可用 CPU 数
    max_workers = min(len(tasks), max(1, os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_meef_perturbation_worker, arg): (arg[0], arg[1])
            for arg in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks),
                           desc="Building MEEF matrix", ncols=100, ascii=" █"):
            try:
                col_idx, side, epe_vec = future.result()
                if side == +1:
                    epe_plus[col_idx] = epe_vec
                else:
                    epe_minus[col_idx] = epe_vec
            except Exception as e:
                fail_col, fail_side = future_to_task[future]
                print(f"\nError computing MEEF for column {fail_col} side {fail_side}: {e}")

    # 中心差分聚合: M[:, col] = (epe_plus - epe_minus) / (2*delta)
    grad = (epe_plus - epe_minus) / (2.0 * delta)  # shape: (num_cps, num_eps)
    grad = np.round(grad, 6)
    M[:] = grad.T  # 转回 (num_eps, num_cps)

    return M
















class  MEEF_Optimizer:
    """
    使用 MEEF 方法的优化器。
    """
    def __init__(self, simulator: LithographySimulator, iterations: int,
                 step_tol: float = 0.0, patience: int = 3) :
        """
        初始化优化器。

        Args:
            simulator: 配置好的 LithographySimulator 实例。
            iterations: 优化的最大迭代次数 (硬上限)。
            step_tol:  位移收敛阈值。当 ‖Δd‖_max < step_tol 连续 patience 次时
                       提前终止。设为 0 (默认) = 不启用, 跑满 iterations。
            patience:  连续满足阈值的次数, 避免单次波动误判。默认 3。
        """
        self.simulator = simulator
        self.iterations = iterations
        self.target_mask = simulator.mask.data.copy()       # 优化目标是原始的二值掩模
        self.render = AntiAliasRenderer(msaa_level=16)      # 使用 MSAA 渲染器
        self.parametric = ParametricDemo(self.simulator.meef.curve_type,
                                         self.target_mask)  # 拟合成为参数化曲线
        self.step_tol = float(step_tol)
        self.patience = int(patience)

    def run(self):
        print('文件保存路径为：',self.simulator.meef.filepath)
        # np.savetxt("LSMmatrix_0525.txt",self.target_mask,fmt='%d',delimiter=' ')
        
        # 1.初始化掩模,LSM优化后的
        initial_mask = self.simulator.meef.lsm_mask
        # self.simulator.update_mask(initial_mask)

        # np.savetxt("目标版图对称通孔.txt", self.target_mask, fmt='%d', delimiter=' ')
        # 测试两种成像方式时使用
        # self.simulator.prepare_for_optimization(method="SOCS")
        # ai_SCOS, wafer_SOCS, intermediates = images_simulation_v2(
        #         mask_spatial=self.target_mask,
        #         resist_params=self.simulator.params.resist,
        #         opt_cache=self.simulator.opt_cache
        #     )
        
        # self.simulator.prepare_for_optimization(method="Abbe")
        # ai_Abbe, wafer_Abbe, intermediates = images_simulation_v2(
        #         mask_spatial=self.target_mask,
        #         resist_params=self.simulator.params.resist,
        #         opt_cache=self.simulator.opt_cache
        #     )
        # print("SOCS的PE:",np.sum((wafer_SOCS-self.target_mask)**2))
        # print("Abbe的PE:",np.sum((wafer_Abbe-self.target_mask)**2))
        # plt.figure(figsize=(10, 10))
        # plt.subplot(2,2,1)

        # plt.imshow(ai_SCOS)
        # plt.colorbar()
        # plt.title('SCOS_ai')
        # plt.subplot(2,2,2)

        # plt.imshow(wafer_SOCS)
        # plt.colorbar()
        # plt.title('SOCS_wafer')
        # plt.subplot(2,2,3)

        # plt.imshow(ai_Abbe)
        # plt.colorbar()
        # plt.title('Abbe_ai')
        # plt.subplot(2,2,4)

        # plt.imshow(wafer_Abbe)
        # plt.colorbar()
        # plt.title('Abbe_wafer')
        # plt.show()
        
        
        
        
        
        
        
       
        # current_mask = self.simulator.get_current_mask_spatial()
        
        # 开始迭代优化,设置当前mask为LSM优化后的
        current_mask = self.simulator.meef.lsm_mask
        # lsm_mask = self.simulator.meef.lsm_mask
        # current_mask = self.simulator.meef.lsm_mask
        # current_mask = np.loadtxt("outputs/opc\CTM和levelset图像\Ls_mask\ls_imagematrix_0011_cpu.txt")
        # 进行水平集优化后的成像
        ai_lsm, wafer_lsm, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )
        # 保存LSM优化后的图像
        _save_images(ai_lsm, wafer_lsm, current_mask, self.target_mask, self.simulator.meef.filepath,"ls_mask", pixel_size=self.simulator.mask.pixel_size)
        
        #计算水平集优化后的误差
        ls_epe, ls_epe_vector = caculate_epe(ai_lsm, self.simulator.params.resist.threshold,self.simulator.meef.eps,self.simulator.meef.num_eps,
                                       self.simulator.meef.weight_meef,add_weight=False)
        ls_pe = np.sum((self.target_mask - wafer_lsm) ** 2)
        np.savetxt(f"{self.simulator.meef.filepath}/target_mask.txt",self.target_mask,fmt='%d',delimiter=' ')
        # print(self.simulator.meef.wepe_caculated.dtype())
        # print(ls_epe_vector.dtype())
        print("M矩阵维数为(需要行大于列)：",self.simulator.meef.num_eps,self.simulator.meef.num_cps)
        ls_wepe = caculate_wepe(self.simulator.meef.wepe_caculated, ls_epe_vector, self.simulator.meef.num_weps)
        print('#######################################################################')
        print(f"LSM后的平均wEPE误差：{ls_wepe/self.simulator.meef.num_weps:.6f}，EPE误差：{ls_epe/self.simulator.meef.num_eps:.6f}，PE误差：{ls_pe:.6f}") 
        
        
        #计算初始参数化曲线拟合后的误差，用于后续的优化表示初始误差
        current_cps = self.simulator.meef.cps
        # current_cps = np.load(r"matrix_0011间隔.npy")
        # ---- 保存初始 (iter=0) B 样条曲线 ----
        _save_bspline_curves(
            parametric=self.parametric,
            new_cps=current_cps,
            iteration_idx=0,
            filepath=self.simulator.meef.filepath,
        )
        # ---- 保存初始 (iter=0) 控制点坐标, 供 CP 轨迹绘制 ----
        _save_cp_history(
            new_cps=current_cps,
            iteration_idx=0,
            filepath=self.simulator.meef.filepath,
        )
        current_mask = self.parametric.render_curve(current_cps) + self.simulator.meef.sraf_mask
        ai_initial, wafer_initial, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )
        # 计算初始EPE误差用于MEEF矩阵的优化
        initial_epe, initial_epe_vector = caculate_epe(ai_initial, self.simulator.params.resist.threshold,self.simulator.meef.eps,self.simulator.meef.num_eps,
                                       self.simulator.meef.weight_meef,add_weight=False)
        
        
        # 记录器
        epe_errors = []
        pe_errors = []
        iterations = []
        wepe_errors = []
        iteration_time = []
        epe_errors.append(ls_epe/self.simulator.meef.num_eps)
        pe_errors.append(ls_pe)
        wepe_errors.append(ls_wepe/self.simulator.meef.num_weps)
        iterations.append(0)
        iteration_time.append(0)
        # ---------- WEPE 最优 ----------
        best_wepe = np.inf
        best_wepe_epe = None
        best_wepe_mask = None
        best_wepe_aerial = None
        best_wepe_wafer = None
        best_wepe_it = None
        best_wepe_pvband = None
        # ---------- EPE 最优 ----------
        best_epe = np.inf
        best_epe_wepe = None
        best_epe_mask = None
        best_epe_aerial = None
        best_epe_wafer = None
        best_epe_it = None
        best_epe__pvband = None
        #绘制初始控制点以及ep点分布
        show_sample_mcps(self.target_mask, self.simulator.meef.initial_mask, self.simulator.meef.cps,
                         self.simulator.meef.eps,self.simulator.meef.filepath, self.simulator.meef.weight_meef)
        
        #计算初始控制点的角平分向量
        current_cps = self.simulator.meef.cps.copy()
        current_vectors = [get_cp_vectors(cps) for cps in current_cps]
        time_start = time.time()
        #构建MEEF矩阵
        # 控制点移动策略:
        #   'bisector' (默认): 沿角平分线方向, 一维位移 (原逻辑)
        #   'xy'              : 分别构建 X / Y 方向 MEEF, 各自求解 Δx / Δy,
        #                       合成出真实的二维移动方向与步长
        
        move_strategy = getattr(self.simulator.meef or self.simulator.meef.move_strategy, "move_strategy", "bisector")


        def _apply_weight(mat):
            """对 MEEF 矩阵按关键 EP 点权重加权 (复用原有规则)。"""
            if self.simulator.meef.pattern_name != "中心对称通孔" or \
               self.simulator.meef.pattern_name != "对角通孔":
                w_cost_array = np.array(self.simulator.meef.weight_meef).flatten()
                mat = mat * w_cost_array[:, np.newaxis]
            else:
                w_cost_array = np.array(self.simulator.meef.weight_meef)
                mat = mat * w_cost_array[:, np.newaxis]
                weight_epe_array = np.array(self.simulator.meef.wepe_caculated)
                mat = mat * weight_epe_array[:, np.newaxis]
            mat[np.where(np.abs(mat) < 1e-3)] = 0
            return mat

        below_tol_count = 0   # 连续满足 ‖Δd‖_max < step_tol 的次数
        actual_iterations = self.iterations
        for iteration_idx in range(1, self.iterations + 1):
            print(f"===== 第 {iteration_idx} 次迭代开始 (move={move_strategy}) =====")
            start_time = time.time()

            if move_strategy == "xy":
                # ---- X/Y 解耦策略: 分别求 Δx, Δy 再合成 ----
                Mx, My = _build_meef_matrix_xy_parallel_threads(
                    simulator=self.simulator,
                    parametric=self.parametric,
                    current_cps=current_cps,
                    target_mask=self.target_mask,
                    sraf_mask=self.simulator.meef.sraf_mask,
                    eps=self.simulator.meef.eps,
                    delta=self.simulator.meef.delta,
                    render=self.render,
                    curve_type=self.simulator.meef.curve_type,
                )
                Mx = _apply_weight(Mx)
                My = _apply_weight(My)
                print("Mx/My 的维度为：", Mx.shape, My.shape)

                e0 = initial_epe_vector
                # X 方向: 解出每个控制点的 Δx
                lam_x = find_turelambadas(Mx, e0)
                delta_x = np.round(truncated_svd_solver(Mx, e0, lam_x), 6)
                # Y 方向: 解出每个控制点的 Δy
                lam_y = find_turelambadas(My, e0)
                delta_y = np.round(truncated_svd_solver(My, e0, lam_y), 6)

                # 合成位移幅度(仅用于日志/保存), 真正更新用 (Δx, Δy)
                delta_d = np.round(np.sqrt(delta_x ** 2 + delta_y ** 2), 6)
                M = Mx  # 供后续 save_iteration_results 仍有矩阵可存

                # 更新控制点: 直接施加二维位移
                new_cps = get_new_cps_xy(current_cps, delta_x, delta_y)
                new_vectors = [get_cp_vectors(cps) for cps in new_cps]
            else:
                # ---- 角平分线策略 (原逻辑) ----
                M = _build_meef_matrix_parallel_threads(
                            simulator=self.simulator,
                            parametric=self.parametric,
                            current_vectors=current_vectors,
                            current_cps=current_cps,
                            target_mask=self.target_mask,
                            sraf_mask=self.simulator.meef.sraf_mask,
                            eps=self.simulator.meef.eps,
                            delta=self.simulator.meef.delta,
                            render=self.render,
                            curve_type=self.simulator.meef.curve_type,
                            pixel_size=self.simulator.mask.pixel_size,
                            )
                M = _apply_weight(M)
                print("M的维度为：", M.shape)
                e0 = initial_epe_vector
                opt_lambadas = find_turelambadas(M, e0)
                delta_d = np.round(truncated_svd_solver(M, e0, opt_lambadas), 6)

                # 更新控制点
                new_vectors = [get_cp_vectors(cps) for cps in current_cps]
                new_cps = get_new_cps(current_cps, delta_d, new_vectors)
            
            #生成新的掩模
            current_mask = self.parametric.render_curve(new_cps) + self.simulator.meef.sraf_mask
            # ---- 保存渲染前的 B 样条曲线 (每次迭代) ----
            _save_bspline_curves(
                parametric=self.parametric,
                new_cps=new_cps,
                iteration_idx=iteration_idx,
                filepath=self.simulator.meef.filepath,
            )
            # ---- 保存当前迭代的控制点坐标, 供 CP 轨迹绘制 ----
            _save_cp_history(
                new_cps=new_cps,
                iteration_idx=iteration_idx,
                filepath=self.simulator.meef.filepath,
            )
            ai_current, wafer_current, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )
            if iteration_idx == 1:
                duration_old = 0
            else:
                duration_old = duration_new
            duration_new = time.time() - start_time + duration_old
            duration_new = int(duration_new)
            
            # 计算当前掩模的成像误差
            current_epe,current_epe_vector = caculate_epe(ai_current, self.simulator.params.resist.threshold,self.simulator.meef.eps,self.simulator.meef.num_eps,
                                       self.simulator.meef.weight_meef,add_weight=False)
            current_wepe = caculate_wepe(self.simulator.meef.wepe_caculated, current_epe_vector, self.simulator.meef.num_weps)
            current_pe = np.round(np.sum((self.target_mask - wafer_current) ** 2),6)
            # 打印本次位移的最大幅度 (帮助监控 TSVD 步长是否在收敛)
            if move_strategy == "xy":
                step_info = (f"|Δx|_max = {np.abs(delta_x).max():.4f}, "
                             f"|Δy|_max = {np.abs(delta_y).max():.4f}, "
                             f"|Δd|_max = {np.abs(delta_d).max():.4f}")
            else:
                step_info = f"|Δd|_max = {np.abs(delta_d).max():.4f}"
            print(f"迭代 {iteration_idx} 次后：平均wEPE误差 = {current_wepe/self.simulator.meef.num_weps:.6f}，"
                  f"EPE误差 = {current_epe/self.simulator.meef.num_eps:.6f},PE误差 = {current_pe:.6f}, {step_info}")
            
            #保存迭代结果
            save_iteration_results(self.simulator.meef.filepath,current_mask,ai_current,wafer_current,delta_d,M,'cividis',
                                   pixel_size=self.simulator.mask.pixel_size)
            epe_errors.append(current_epe/self.simulator.meef.num_eps)
            pe_errors.append(current_pe)
            iterations.append(iteration_idx)
            wepe_errors.append(current_wepe/self.simulator.meef.num_weps)
            iteration_time.append(duration_new)
            
            if current_wepe < best_wepe:
                best_wepe = current_wepe
                best_wepe_epe = current_epe
                best_wepe_mask, best_wepe_aerial, best_wepe_wafer = current_mask.copy(), ai_current.copy(), wafer_current.copy()
                best_wepe_it = iteration_idx
                best_wepe_cps = current_cps.copy()
                
            if current_epe < best_epe:
                best_epe = current_epe
                best_epe_wepe = current_wepe
                best_epe_mask, best_epe_aerial, best_epe_wafer = current_mask.copy(), ai_current.copy(), wafer_current.copy()
                best_epe_it = iteration_idx
                best_epe_cps = current_cps.copy()
            
            #更新当前所需的参数
            initial_epe_vector = current_epe_vector.copy()
            current_cps = new_cps.copy()
            current_vectors = new_vectors.copy()

            # ---- 位移收敛准则检查 (基于 TSVD 步长 Δd) ----
            if self.step_tol > 0:
                step_max = float(np.abs(delta_d).max())
                if step_max < self.step_tol:
                    below_tol_count += 1
                    print(f"  [convergence] ‖Δd‖_max={step_max:.6f} < "
                          f"tol={self.step_tol} ({below_tol_count}/{self.patience})")
                    if below_tol_count >= self.patience:
                        print(f"===== 提前终止: 连续 {self.patience} 次 "
                              f"‖Δd‖_max < {self.step_tol}, 第 {iteration_idx} 次迭代 =====")
                        actual_iterations = iteration_idx
                        break
                else:
                    below_tol_count = 0
        time_end = time.time()
        print(f"[summary] 实际迭代次数: {actual_iterations} / {self.iterations}")
        print(f"迭代总耗时：{time_end - time_start:.2f}秒")
        #保存结果
        save_excel(self.simulator.meef.up_filename, 
                   self.simulator.meef.second_filename, 
                   self.simulator.meef.curve_type,
                   self.simulator.meef.filename,
                   iterations, 
                   pe_errors, 
                   epe_errors, 
                   wepe_errors,
                   iteration_time)    
        print("===== MEEF 曲线优化完成 =====")
        drawcostcurve(epe_errors, iterations, f'{self.simulator.meef.filepath}/curves/error/EPE_error.png', 'EPE_error', 'EPE')
        drawcostcurve(wepe_errors, iterations, f'{self.simulator.meef.filepath}/curves/error/wEPE_error.png', 'wEPE_error', 'wEPE')
        drawcostcurve(pe_errors, iterations, f'{self.simulator.meef.filepath}/curves/error/PE_error.png', 'PE_error', 'PE')
        # ---- 画出 B 样条曲线的演化对比图 (iter=0 vs best 几次) ----
        _plot_bspline_evolution(
            filepath=self.simulator.meef.filepath,
            target_mask=self.target_mask,
        )
        save_dir = Path(self.simulator.meef.filepath)   # 目录
        # ================== WEPE 最优结果 ==================
        save_path = save_dir / "best_wepe_result.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"迭代过程中WEPE最优为第 {best_wepe_it} 次迭代\n"
                f"EPE误差为 {best_wepe_epe / self.simulator.meef.num_eps}\n"
                f"WEPE误差为 {best_wepe / self.simulator.meef.num_weps}\n"
                
            )
        with open(
            f"{self.simulator.meef.filepath}/wEPE最优的控制点坐标.txt",
            "w"
        ) as f:
            for row in best_wepe_cps:
                f.write(" ".join(map(str, row)) + "\n")

        _ps = self.simulator.mask.pixel_size
        _sraf_cps = getattr(self.simulator.meef, "sraf_cps", None)

        # wafer: 带 nm 坐标轴 + colorbar (统一风格)
        save_matrix_nm(
            best_wepe_wafer,
            f"{self.simulator.meef.filepath}/wEPE最优wafer.png",
            pixel_size=_ps, cmap="cividis", dpi=600,
        )
        # 保留红色目标轮廓叠加版供参考
        highlight_contour_red(
            self.target_mask, best_wepe_wafer,
            save_path=f"{self.simulator.meef.filepath}/wEPE最优wafer_contour.png",
        )
        # mask (像素化): 带 nm 坐标轴 + colorbar
        save_matrix_nm(
            best_wepe_mask,
            f"{self.simulator.meef.filepath}/wEPE最优mask.png",
            pixel_size=_ps, cmap="cividis",
        )
        # 新增: wEPE 最优 CP 拟合成的 B-spline 曲线掩模
        save_bspline_curve_mask(
            cps_list=best_wepe_cps,
            sraf_cps_list=_sraf_cps,
            target_mask=self.target_mask,
            save_path=f"{self.simulator.meef.filepath}/wEPE最优bspline.png",
            curve_type=self.simulator.meef.curve_type,
            pixel_size=_ps,
            title="wEPE optimal B-spline mask",
        )
        # ================== EPE 最优结果 ==================
        save_path = save_dir / "best_epe_result.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"迭代过程中EPE最优为第 {best_epe_it} 次迭代\n"
                f"EPE误差为 {best_epe / self.simulator.meef.num_eps}\n"
                f"WEPE误差为 {best_epe_wepe / self.simulator.meef.num_weps}\n"
                f"LSM优化后的WEPE误差为 {ls_wepe / self.simulator.meef.num_weps}\n"
                F"EPE误差为 {ls_epe / self.simulator.meef.num_eps}\n"
            )

        with open(
            f"{self.simulator.meef.filepath}/EPE最优的控制点坐标.txt",
            "w"
        ) as f:
            for row in best_epe_cps:
                f.write(" ".join(map(str, row)) + "\n")

        # wafer: 带 nm 坐标轴 + colorbar
        save_matrix_nm(
            best_epe_wafer,
            f"{self.simulator.meef.filepath}/EPE最优wafer.png",
            pixel_size=_ps, cmap="cividis", dpi=600,
        )
        highlight_contour_red(
            self.target_mask, best_epe_wafer,
            save_path=f"{self.simulator.meef.filepath}/EPE最优wafer_contour.png",
        )
        save_matrix_nm(
            best_epe_mask,
            f"{self.simulator.meef.filepath}/EPE最优mask.png",
            pixel_size=_ps, cmap="cividis",
        )
        # 新增: EPE 最优 CP 拟合成的 B-spline 曲线掩模
        save_bspline_curve_mask(
            cps_list=best_epe_cps,
            sraf_cps_list=_sraf_cps,
            target_mask=self.target_mask,
            save_path=f"{self.simulator.meef.filepath}/EPE最优bspline.png",
            curve_type=self.simulator.meef.curve_type,
            pixel_size=_ps,
            title="EPE optimal B-spline mask",
        )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

           