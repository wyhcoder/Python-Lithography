from matplotlib import pyplot as plt
import math
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
def show_sample_mcps(target,initial_mask1,initial_cps,initial_eps,file_path, w_cost):
        import numpy as np
import matplotlib.pyplot as plt
import os


def show_sample_mcps(target,
                     initial_mask1,
                     initial_cps,
                     initial_eps,
                     file_path,
                     w_cost):
    """
    保存并显示 EP / CP 点可视化结果

    输出文件：
        ep_points.png
        cp_points.png
        ep_points.txt
        cp_points.txt
    """

    os.makedirs(file_path, exist_ok=True)

    # =====================================================
    # 1️⃣ EP mask 构建
    # =====================================================
    ep_mask = target.copy().astype(float)

    # 所有 EP 赋值为 4
    for ep in initial_eps:
        y, x = map(int, ep)
        ep_mask[y, x] = 4

    # 选出 w_cost == 4 的 EP
    w_cost_array = np.array(w_cost).flatten()
    idx = (w_cost_array == 4)

    if np.any(idx):
        selected_eps = initial_eps[idx, :]
        for ep in selected_eps:
            y, x = map(int, ep)
            ep_mask[y, x] = 20

    # =====================================================
    # 2️⃣ CP mask 构建
    # =====================================================
    cp_mask = initial_mask1.copy().astype(float)

    for cps in initial_cps:
        for cp in cps:
            y, x = map(int, cp)
            cp_mask[y, x] = 6

    # =====================================================
    # 3️⃣ 分别保存图片
    # =====================================================
    plt.imsave(
        os.path.join(file_path, "w_ep_points.png"),
        ep_mask,
        cmap="plasma",
        vmin=0,
        vmax=21
    )

    plt.imsave(
        os.path.join(file_path, "cp_points.png"),
        cp_mask,
        cmap="plasma"
    )

    # =====================================================
    # 4️⃣ 分别保存 TXT
    # =====================================================
    np.savetxt(
        os.path.join(file_path, "ep_points.txt"),
        ep_mask,
        delimiter=" ",
        fmt="%d"
    )

    np.savetxt(
        os.path.join(file_path, "cp_points.txt"),
        cp_mask,
        delimiter=" ",
        fmt="%d"
    )

    # =====================================================
    # 5️⃣ 同窗口显示
    # =====================================================
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("EP Points")
    plt.imshow(ep_mask, cmap="plasma", vmin=0, vmax=21)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("CP Points")
    plt.imshow(cp_mask, cmap="plasma")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
 

class Control_vector:
        def __init__(self, cp_point, vector):
            self.cp = cp_point
            self.vector =   vector    

def get_cp_vectors(control_points):
        """
        获取当前控制点的角平分线向量
        """
        control_vectors_class_list = []
        control_points = [np.array(p, dtype=np.float32) for p in control_points]
        for i in range(len(control_points)):
            Q_prev = np.array(control_points[i-1])
            Q_current = np.array(control_points[i])
            Q_next = np.array(control_points[(i+1)%len(control_points)])
            v1 = Q_prev - Q_current
            v2 = Q_next - Q_current
            # 单位向量
            v1_unit = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_unit = v2 / (np.linalg.norm(v2) + 1e-8)
            
            bisector = v1_unit + v2_unit
            bisector_norm = np.linalg.norm(bisector)

            if bisector_norm < 1e-6:
                # 特殊情况（180度） - 用 v2 的法线
                bisector = np.array([-v2_unit[1], v2_unit[0]])
            else:
                bisector = bisector / bisector_norm
                
            # 外法线方向（用当前边）
            edge = Q_next - Q_current
            normal = np.array([-edge[1], edge[0]])  # 向外法线（逆时针轮廓）

            # 若角平分线朝内，则翻转方向
            if np.dot(bisector, normal) > 0:
                bisector = -bisector
            # print('bisector', bisector)
            control_vectors_class_list.append(
            Control_vector(
                Q_current,
                bisector
            )
        )
        return control_vectors_class_list
    
def highlight_contour_red(target1, wi, save_path="wi_red.png"):
        """
        将 wi 用 inferno 色图渲染为 RGB，并在其上叠加 target 的红色轮廓后保存 PNG。

        参数
        ----
        target1 : np.ndarray  # 目标版图，0/1 或灰度
        wi      : np.ndarray  # 需要着色的矩阵 (灰度, 任意范围, 内部会归一化到 [0,1])
        save_path : str       # 输出路径
        """

        # 1) 二值化 target，提轮廓
        target = np.asarray(target1).copy()
        target_bin = (target > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            target_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 2) 用 inferno 色图渲染 wi
        wi_arr = np.asarray(wi).astype(np.float32)
        if wi_arr.ndim != 2:
            # 已是彩色, 转灰度后再上色, 保证统一 inferno 风格
            wi_arr = wi_arr.mean(axis=-1)

        wi_min, wi_max = float(wi_arr.min()), float(wi_arr.max())
        if wi_max - wi_min > 1e-12:
            wi_norm = (wi_arr - wi_min) / (wi_max - wi_min)
        else:
            wi_norm = np.zeros_like(wi_arr, dtype=np.float32)

        # matplotlib 色图 -> RGBA (float, 0-1)
        cmap_rgba = plt.get_cmap("cividis")(wi_norm)
        wi_rgb = (cmap_rgba[..., :3] * 255.0).astype(np.uint8)
        # OpenCV 用 BGR
        wi_bgr = cv2.cvtColor(wi_rgb, cv2.COLOR_RGB2BGR)

        # 3) 在 inferno 图上画红色 target 轮廓
        cv2.drawContours(wi_bgr, contours, -1, color=(0, 0, 255), thickness=1)

        # 4) 保存 PNG
        plt.imsave(
            save_path,
            cv2.cvtColor(wi_bgr, cv2.COLOR_BGR2RGB),
            dpi=600,
        )

def move_single_control_point(control_point, delta,vector):
        
        if delta > 0: #向外为正
            new_control_point = control_point + delta * (vector)
            
        if delta < 0: #向内为负
            new_control_point = control_point + delta * (vector)
        
        # return np.round(new_control_point,4)
        return  np.round(np.array(new_control_point, dtype=np.float64), 4)

def move_single_control_point_X(control_point, delta):
    """
    仅沿 x 方向平移控制点 (delta>0 向 +x, delta<0 向 -x)。
    """
    cp = np.asarray(control_point, dtype=np.float64)
    new_control_point = cp + delta * np.array([1.0, 0.0])
    return np.round(new_control_point, 4)


def move_single_control_point_Y(control_point, delta):
    """
    仅沿 y 方向平移控制点 (delta>0 向 +y, delta<0 向 -y)。
    """
    cp = np.asarray(control_point, dtype=np.float64)
    new_control_point = cp + delta * np.array([0.0, 1.0])
    return np.round(new_control_point, 4)


def get_new_cps_xy(old_cps, delta_x, delta_y):
    """
    用 X/Y 两个方向各自解出的位移量合成新的控制点。

    与 `get_new_cps` (沿角平分线一维位移) 不同, 这里每个控制点同时
    施加 (Δx, Δy), 合成出真实的二维移动向量。

    参数:
        old_cps : list[ndarray(N_i, 2)]  原始控制点 (按 contour 分组)
        delta_x : 1D array, 长度 = 总控制点数, X 方向位移 (按展平顺序)
        delta_y : 1D array, 长度 = 总控制点数, Y 方向位移 (按展平顺序)

    返回:
        list[ndarray(N_i, 2)] 新控制点, 结构与 old_cps 一致。
    """
    delta_x = np.asarray(delta_x, dtype=np.float64).flatten()
    delta_y = np.asarray(delta_y, dtype=np.float64).flatten()

    all_new_cps = []
    m = 0
    for contour in old_cps:
        new_contour = []
        for cp in contour:
            moved = np.asarray(cp, dtype=np.float64) + np.array(
                [delta_x[m], delta_y[m]]
            )
            new_contour.append(np.round(moved, 4))
            m += 1
        all_new_cps.append(np.array(new_contour, dtype=np.float64))
    return all_new_cps


def compute_L_curve(A, b, lambdas):
        """
        计算一组正则化参数 lambda 对应的 L-curve 数据点。

        L-curve 的横纵坐标通常定义为：
            - 残差范数 ||Ax - b||
            - 解范数 ||x||

        这里针对每一个 lambda，求解 Tikhonov 正则化问题：
            x = argmin ||Ax - b||^2 + lambda * ||x||^2

        等价的正规方程形式为：
            (A^T A + lambda I)x = A^T b

        参数：
            A : ndarray, shape (m, n)
                系数矩阵
            b : ndarray, shape (m,) 或 (m,1)
                右端项
            lambdas : ndarray
                一组候选正则化参数

        返回：
            res_norms : ndarray
                每个 lambda 对应的残差范数 ||Ax - b||
            x_norms : ndarray
                每个 lambda 对应的解范数 ||x||
        """
        m, n = A.shape
        res_norms = []
        x_norms = []
    
        for lam in lambdas:
        # 计算Tikhonov正则化解
            ATA = A.T @ A
            regularization = lam * np.eye(n)
            x = np.linalg.solve(ATA + regularization, A.T @ b)
        
        # 计算残差范数和解范数
            res_norm = np.linalg.norm(A @ x - b)
            x_norm = np.linalg.norm(x)
            res_norms.append(res_norm)
            x_norms.append(x_norm)
    
        return np.array(res_norms), np.array(x_norms)

def find_optimal_lambda(lambdas, res_norms, x_norms):    
        """
        根据 L-curve 的曲率，寻找最优正则化参数 lambda。

        思路：
            1. 将 L-curve 转换到对数坐标系：
                x = log10(res_norms)
                y = log10(x_norms)
            2. 计算曲线的数值一阶导和二阶导
            3. 利用曲率公式，找到曲率最大的点
            4. 该点对应的 lambda 作为最优正则化参数

        曲率公式：
            kappa = |y''| / (1 + (y')^2)^(3/2)

        参数：
            lambdas : ndarray
                候选正则化参数
            res_norms : ndarray
                对应的残差范数
            x_norms : ndarray
                对应的解范数

        返回：
            optimal_lam : float
                曲率最大点对应的最优 lambda
        """
        
        log_res = np.log10(res_norms)
        log_x = np.log10(x_norms)
    
    # 计算曲率 (数值微分)
        d1 = np.gradient(log_x, log_res)      # 一阶导数 dy/dx
        d2 = np.gradient(d1, log_res)          # 二阶导数 d²y/dx²
        curvature = np.abs(d2) / (1 + d1**2)**1.5  # 曲率公式
    
    # 找到曲率最大的点（排除边界点）
        max_idx = np.argmax(curvature[1:-1]) + 1
        optimal_lam = lambdas[max_idx]
        return optimal_lam

def find_turelambadas(M, e0):
        """
            利用 L-curve 方法自动选择 Tikhonov 正则化参数 lambda。

        这里求解的问题可写为：
            min ||M x + e0||^2 + lambda ||x||^2

        将其写成标准形式：
            A = M
            b = -e0

        然后通过扫描一系列 lambda：
            1. 计算每个 lambda 对应的残差范数和解范数
            2. 构造 L-curve
            3. 选择曲率最大的点作为最优 lambda

        参数：
            M : ndarray
                系数矩阵
            e0 : ndarray
                初始误差项

        返回：
            optimal_lam : float
                通过 L-curve 方法选取的最优正则化参数
            
        """
        A = M
        b = -e0
        lambdas = np.logspace(-6, 2, 100)
        res_norms, x_norms = compute_L_curve(A, b, lambdas)
        optimal_lam = find_optimal_lambda(lambdas, res_norms, x_norms)
        return optimal_lam

def truncated_svd_solver(M, e, λ, energy_threshold=0.96, plot_singular_values=False):
        # 计算完整SVD
        U, S, VT = np.linalg.svd(M, full_matrices=False)
    
        # 计算截断参数 k（能量占比法）,np.cumsum(S**2)计算前k个奇异值的平方和，np.sum(S**2)计算所有奇异值的平方和
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        k = np.argmax(cumulative_energy >= energy_threshold) + 1

        #动态调整截断参数 k
       
        # 可视化奇异值衰减
        if plot_singular_values:
            plt.plot(S, 'o-', label='Singular Values')
            plt.axvline(x=k-1, color='r', linestyle='--', label=f'Truncation at k={k}')
            plt.xlabel('Index')
            plt.ylabel('Magnitude')
            plt.title('Singular Value Decay')
            plt.legend()
            plt.show()
    
       
        s = S[:k]
        scale = s / (s**2 + λ)
        D = np.diag(scale)
        e = -e.reshape(-1, 1)
        Δx = VT[:k, :].T @ D @ U[:, :k].T @ e

        return Δx.flatten()


def get_new_cps(old_cps, delta_d,cps_vectors) :
        """ 
        根据delta_d和cps_vectors计算新的cps
        """
        new_cps = []
        all_new_cps = []
        m  = 0
        for i in range(len(old_cps)):
            for j in range(len(old_cps[i])):
                new_cps.append(old_cps[i][j] + delta_d[m] * cps_vectors[i][j].vector)
                m += 1
            all_new_cps.append(new_cps)
            new_cps = []
        return all_new_cps


def _save_with_colorbar(image, save_path, cmap, dpi: int = 300, title: str = None):
        """用 imshow + colorbar 保存一张二维数组的可视化图，自带颜色条。

        - 关闭坐标轴；
        - 输出 dpi 由参数控制；
        - 保存后立即关闭 figure 防内存累积。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.asarray(image)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(arr, cmap=cmap)
        ax.set_axis_off()
        if title:
            ax.set_title(str(title))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_matrix_nm(matrix, save_path, pixel_size, cmap="cividis",
                   dpi: int = 600, title: str = None,
                   cbar_label: str = None):
    """保存二维矩阵, 坐标轴用物理单位 nm, 以矩阵中心为原点。

    坐标换算 (中心对齐):
        coord(i) = (i - (N - 1) / 2) * pixel_size
    例如 N=257, pixel_size=4 时, 范围为 -512 nm ~ +512 nm。

    参数:
        matrix     : 2D array, 待保存的矩阵 (mask / aerial / wafer / M 等)。
        save_path  : 输出图片路径 (含文件名)。
        pixel_size : 每像素物理尺寸 (nm)。
        cmap       : colormap, 默认 cividis。
        dpi        : 输出分辨率。
        title      : 可选标题。
        cbar_label : 可选 colorbar 标签。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(matrix)
    h, w = arr.shape[:2]
    ps = float(pixel_size)

    # 中心对齐的物理坐标范围 (nm)
    x_min = -(w - 1) / 2.0 * ps
    x_max = (w - 1) / 2.0 * ps
    y_min = -(h - 1) / 2.0 * ps
    y_max = (h - 1) / 2.0 * ps
    # 用 origin='upper' (matplotlib 默认, 行0在顶部, 与数据存储一致),
    # extent 的 y 方向写成 [top, bottom] = [y_max, y_min], 使 y 轴刻度
    # 上正下负, 且图像不会上下翻转。
    extent = [x_min - ps / 2.0, x_max + ps / 2.0,
              y_max + ps / 2.0, y_min - ps / 2.0]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if title:
        ax.set_title(str(title))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_bspline_curve_mask(
    cps_list,
    sraf_cps_list,
    target_mask,
    save_path,
    curve_type: str = "BS",
    pixel_size: float = None,
    smoothing: float = 0.3,
    num_points: int = 200,
    dpi: int = 300,
    title: str = None,
    show_cps: bool = True,
    cp_color: str = "red",
    cp_size: float = 14.0,
):
    """把优化最优的 CP 拟合成 B-spline (或其他参数化) 曲线保存为图像。

    与像素化掩模不同, 本函数画的是矢量曲线, 便于与目标轮廓叠加对比;
    坐标轴优先使用 nm 单位 (提供 pixel_size 时) 以保持与 wafer/mask 统一。

    参数
    ----
    cps_list : list of ndarray
        主图形每个轮廓的控制点, 形状 (K_i, 2), 每行 [y, x] (像素坐标)。
    sraf_cps_list : list of ndarray or None
        SRAF 每个 block 的控制点, 形状 (K_j, 2)。None 表示不叠加 SRAF。
    target_mask : ndarray
        目标版图 (H, W), 用于画灰色轮廓作参考。
    save_path : str/Path
        输出 PNG 路径。
    curve_type : str
        参数化类型 ("BS" B-spline, 或 ParametricDemo 支持的其他类型)。
    pixel_size : float or None
        每像素物理尺寸 nm, 给定后使用中心对齐 nm 坐标轴 (与 save_matrix_nm 一致);
        None 时用像素坐标。
    smoothing : float
        B-spline 平滑参数 (默认 0.3, 与优化过程使用一致)。
    num_points : int
        每条曲线采样点数 (默认 200)。
    show_cps : bool
        是否在曲线上叠加控制点散点标记, 默认 True。
    cp_color : str
        控制点标记颜色, 默认 "red"。
    cp_size : float
        控制点标记大小 (matplotlib scatter s 参数), 默认 14.0。
    """
    # 延迟 import 避免循环依赖
    from utils_model.demo_parametric import ParametricDemo

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    target_mask = np.asarray(target_mask)
    H, W = target_mask.shape[:2]
    parametric = ParametricDemo(curve_type, target_mask)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 坐标换算: 像素 -> nm (与 save_matrix_nm 保持一致)
    if pixel_size is not None:
        ps = float(pixel_size)
        # extent 用于 target_mask 底图
        x_min = -(W - 1) / 2.0 * ps
        x_max = (W - 1) / 2.0 * ps
        y_min = -(H - 1) / 2.0 * ps
        y_max = (H - 1) / 2.0 * ps
        extent = [x_min - ps / 2.0, x_max + ps / 2.0,
                  y_max + ps / 2.0, y_min - ps / 2.0]
        # 像素 -> nm 的转换 (x_pix -> x_nm)
        def px2nm_x(x_pix): return (x_pix - (W - 1) / 2.0) * ps
        def px2nm_y(y_pix): return (y_pix - (H - 1) / 2.0) * ps
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    else:
        extent = None
        def px2nm_x(x_pix): return x_pix
        def px2nm_y(y_pix): return y_pix
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")

    # 目标版图轮廓 (灰色, 半透明)
    if pixel_size is not None:
        # 用 extent 定位到 nm 坐标系
        ys_pix = np.arange(H)
        xs_pix = np.arange(W)
        X_nm, Y_nm = np.meshgrid(px2nm_x(xs_pix), px2nm_y(ys_pix))
        ax.contour(X_nm, Y_nm, target_mask, levels=[0.5],
                   colors="gray", linewidths=1.5, alpha=0.7)
    else:
        ax.contour(target_mask, levels=[0.5],
                   colors="gray", linewidths=1.5, alpha=0.7)

    def _draw_curves(cps_iter, color, lw, label=None, cp_marker_size=None):
        """把一组 CP 用 parametric 拟合成闭合曲线, 逐条绘制;
        并在曲线上叠加控制点散点。"""
        first = True
        for cps in cps_iter:
            cps_arr = np.asarray(cps)
            if cps_arr.ndim != 2 or cps_arr.shape[0] < 2:
                continue
            if cps_arr.shape[0] < 4:
                # CP 太少, 直接用折线闭合
                xs = np.append(cps_arr[:, 1], cps_arr[0, 1])
                ys = np.append(cps_arr[:, 0], cps_arr[0, 0])
            else:
                try:
                    curves = parametric.b_spile(
                        [cps_arr], smoothing, num_points=num_points
                    )
                    cv = np.asarray(curves[0])  # (N, 2) [y, x]
                    xs = np.append(cv[:, 1], cv[0, 1])
                    ys = np.append(cv[:, 0], cv[0, 0])
                except Exception:
                    xs = np.append(cps_arr[:, 1], cps_arr[0, 1])
                    ys = np.append(cps_arr[:, 0], cps_arr[0, 0])
            ax.plot(px2nm_x(xs), px2nm_y(ys),
                    color=color, lw=lw, alpha=0.9,
                    label=(label if first else None))

            # 叠加控制点散点 (红色)
            if show_cps:
                s = cp_marker_size if cp_marker_size is not None else cp_size
                # 边框宽度随 marker 大小缩放, 小点用更细的边框避免糊掉
                edge_lw = max(0.1, min(0.4, s / cp_size * 0.4))
                ax.scatter(px2nm_x(cps_arr[:, 1]), px2nm_y(cps_arr[:, 0]),
                           s=s, c=cp_color, marker="o",
                           edgecolors="white", linewidths=edge_lw,
                           zorder=5)
            first = False

    # 主图形曲线 + CP
    _draw_curves(cps_list, color="blue", lw=2.0,
                 label="Main B-spline", cp_marker_size=cp_size)

    # SRAF 曲线 + CP (显著减小, 避免密集 CP 挤成一片)
    if sraf_cps_list is not None and len(sraf_cps_list) > 0:
        _draw_curves(sraf_cps_list, color="blue", lw=1.2,
                     label="SRAF B-spline",
                     cp_marker_size=max(cp_size * 0.25, 2.5))

    # 单独给红色 CP 加一个 legend 条目
    if show_cps:
        ax.scatter([], [], s=cp_size, c=cp_color, marker="o",
                   edgecolors="white", linewidths=0.4,
                   label="Control points")

    ax.set_aspect("equal")
    if pixel_size is not None:
        ax.set_xlim(px2nm_x(0) - ps / 2.0, px2nm_x(W - 1) + ps / 2.0)
        ax.set_ylim(px2nm_y(H - 1) + ps / 2.0, px2nm_y(0) - ps / 2.0)
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    ax.grid(True, ls=":", alpha=0.3)
    if title:
        ax.set_title(str(title))
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_width_results(
        file_path,current_mask,current_SRAFs,aI,wI,width, mask_color,
        pixel_size=None):
        base_dir = Path(file_path)

        # ===== 统一 & 合理的目录结构 =====
        sub_dirs = ['current_mask','current_mask_txt','current_SRAFs', 'current_SRAFs_txt','ai','ai_txt','wi','wi_txt','width','error','sraf_layer']

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
        if pixel_size is not None:
            save_matrix_nm(current_mask,  mask_png_path, pixel_size, cmap=mask_color)
            save_matrix_nm(current_SRAFs, sraf_png_path, pixel_size, cmap=mask_color)
            save_matrix_nm(aI,            ai_png_path,   pixel_size, cmap=mask_color)
            save_matrix_nm(wI,            wi_png_path,   pixel_size, cmap=mask_color)
        else:
            _save_with_colorbar(current_mask,  mask_png_path, cmap=mask_color)
            _save_with_colorbar(current_SRAFs, sraf_png_path, cmap=mask_color)
            _save_with_colorbar(aI,            ai_png_path,   cmap=mask_color)
            _save_with_colorbar(wI,            wi_png_path,   cmap=mask_color)


def save_iteration_results(file_path,  gray_mask, aI, wI, delta_d, M, mask_color,
                           pixel_size=None):
    """
    保存当前迭代结果（覆盖模式）。
    每次调用都会覆盖旧文件，只保留最新状态。

    pixel_size: 每像素物理尺寸 (nm)。给定时, 图像坐标轴以 nm 为单位、
                以矩阵中心为原点; 为 None 时退回无坐标轴的旧风格。
    """
    base_dir = Path(file_path)

    # ===== 方案 A: 按用途两层分组 =====
    #   figures/  所有可视化 PNG (文件名直接用类型名)
    #   data/     所有数值 txt
    fig_dir = base_dir / 'figures'
    data_dir = base_dir / 'data'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 图像路径
    gray_mask_path = fig_dir / 'gray_mask.png'
    ai_path = fig_dir / 'ai.png'
    wi_path = fig_dir / 'wi.png'
    M_path = fig_dir / 'M.png'

    # 文本/数据路径 (命名统一为小写类型名)
    delta_d_path = data_dir / 'delta_d.txt'
    M_txt_path = data_dir / 'M.txt'
    gray_txt_path = data_dir / 'gray_mask.txt'
    wI_txt_path = data_dir / 'wi.txt'

    # 保存文本数据 (自动覆盖)
    np.savetxt(str(delta_d_path), delta_d, fmt='%f', delimiter=',')
    np.savetxt(str(M_txt_path), M, fmt='%.5f')
    np.savetxt(str(gray_txt_path), gray_mask, fmt='%.5f')
    np.savetxt(str(wI_txt_path), wI, fmt='%.5f')

    # 保存带颜色条的图像 (给定 pixel_size 则带 nm 坐标轴)
    if pixel_size is not None:
        save_matrix_nm(gray_mask, gray_mask_path, pixel_size, cmap=mask_color, dpi=600)
        save_matrix_nm(aI,        ai_path,        pixel_size, cmap=mask_color, dpi=600)
        save_matrix_nm(wI,        wi_path,        pixel_size, cmap=mask_color, dpi=600)
        save_matrix_nm(M,         M_path,         pixel_size, cmap=mask_color, dpi=600)
    else:
        _save_with_colorbar(gray_mask, gray_mask_path, cmap=mask_color, dpi=600)
        _save_with_colorbar(aI,        ai_path,        cmap=mask_color, dpi=600)
        _save_with_colorbar(wI,        wi_path,        cmap=mask_color, dpi=600)
        _save_with_colorbar(M,         M_path,         cmap=mask_color, dpi=600)

    # (可选) 如果你想知道当前是第几次，可以在控制台打印，但不体现在文件名上
    # print(f"Iteration {it}: Results overwritten.")
def save_excel(up_filename, second_filename, curve_type,filename,iteration_idx, current_pe, current_epe, current_w_epe,time, file_path=None):
        # 构建 DataFrame
        df = pd.DataFrame({
            'Iteration': iteration_idx,
            'PE': current_pe,
            'EPE': current_epe,
            'wEPE': current_w_epe,
            'time' : time
            })
        df = df.round({'PE': 6, 'EPE': 6, 'wEPE': 6})

        # 优先使用调用方给定目录；否则按历史路径推断并自动建目录
        if file_path:
            out_dir = Path(file_path)
        else:
            out_dir = Path(up_filename) / second_filename / curve_type / filename
            midcheck_dir = Path(up_filename) / second_filename / "中期检查" / filename
            if (not out_dir.exists()) and midcheck_dir.exists():
                out_dir = midcheck_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_dir / 'errors.xlsx', index=False)


def drawcostcurve(PE_list, IT, filepath, title, type):
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        min_value = min(PE_list)
        min_index = PE_list.index(min_value)

        # 用独立 figure, 防止多次调用时多条曲线叠加到同一张图上;
        # 同时避免 plt.show() 阻塞主进程 (批量绘图时尤其重要).
        fig, ax = plt.subplots()
        if type == 'PE':
            ax.plot(IT, PE_list, linestyle='-.', color='r', label='PE_error')
        if type == 'EPE':
            ax.plot(IT, PE_list, linestyle='-.', color='r', label='EPE_error')
        if type == 'wEPE':
            ax.plot(IT, PE_list, linestyle='-.', color='r', label='wEPE_error')
        if type == 'pvband':
            ax.plot(IT, PE_list, linestyle='-.', color='r', label='pvband')
        x_min = IT[min_index]
        y_min = PE_list[min_index]
        y_offset = (max(PE_list) - min(PE_list)) * 0.05  # 动态偏移量

        ax.annotate(f'Min: {min_value:.2f}',
                    xy=(x_min, y_min),
                    xytext=(x_min, y_min + y_offset),
                    arrowprops=dict(facecolor='black', shrink=0.05, linewidth=0.5),
                    ha='center', va='bottom')

        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=600)
        plt.close(fig)

def show_images(self,images, titles=None, ncols=3, cmap='gray'):
    """
    同时显示多张 2D 图像（自动排版）

    参数
    ----
    images : list[np.ndarray]
        图像列表，每个是 2D array
    titles : list[str] or None
        每张图的标题
    ncols : int
        每行显示的图像数量
    cmap : str
        颜色映射
    """
    n = len(images)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < n:
            im = ax.imshow(images[i], cmap=cmap)
            if titles:
                ax.set_title(titles[i])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
def _save_images( aI, wI, mask, target, file_path,name, pixel_size=None):
        mask_color = 'cividis'
        base_dir = Path(file_path)
        aI_path = base_dir / f"{name}_AI.png"
        wI_path = base_dir / f"{name}_WI.png"
        mask_path = base_dir / f"{name}_mask.png"
        # AI / mask: 给定 pixel_size 则带 nm 坐标轴; wafer 仍用红轮廓叠加
        if pixel_size is not None:
            save_matrix_nm(np.asarray(aI),   aI_path,   pixel_size, cmap=mask_color, dpi=600)
            save_matrix_nm(np.asarray(mask), mask_path, pixel_size, cmap=mask_color, dpi=600)
            save_matrix_nm(np.asarray(wI),   wI_path,   pixel_size, cmap=mask_color, dpi=600)
        else:
            plt.imsave(str(aI_path), np.asarray(aI), cmap=mask_color, dpi=600)
            plt.imsave(str(mask_path), np.asarray(mask), cmap=mask_color, dpi=600)
        highlight_contour_red(target,wI,wI_path)
        # plt.imsave(str(wI_path), np.asarray(wI), cmap=mask_color, dpi=600)