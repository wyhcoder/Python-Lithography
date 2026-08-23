import numpy as np

def evolve_kappa_gradient(phi: np.ndarray, dx: float, dy: float, b: float) -> np.ndarray:
    """
    计算基于曲率的演化量。
    使用 numpy.gradient 实现，代码简洁、高效且数值上更稳健。

    参数:
        phi: 2D 水平集函数。
        dx: x方向网格间距。
        dy: y方向网get_dt_normal_kappa.m间距。
        b: 曲率（正则化）系数。
        
    返回:
        delta: 曲率演化量 (b * κ * |∇φ|)。
    """
    # 1. 计算一阶导数 (phi_y, phi_x)
    # np.gradient 返回的顺序是 (dy, dx)，对应轴 (0, 1)
    phi_y, phi_x = np.gradient(phi, dy, dx, edge_order=2)
    
    # 2. 计算二阶导数
    # 计算 phi_x 的梯度，得到 (phi_xy, phi_xx)
    phi_xy, phi_xx = np.gradient(phi_x, dy, dx, edge_order=2)
    
    # 计算 phi_y 的梯度，得到 (phi_yy, phi_yx)
    phi_yy, _ = np.gradient(phi_y, dy, dx, edge_order=2)
    # 根据克莱罗定理, 对于平滑函数 phi_xy = phi_yx，所以我们只需要计算一次。

    # 3. 计算曲率演化项 κ * |∇φ|
    # 公式: (phi_xx*phi_y**2 - 2*phi_y*phi_x*phi_xy + phi_yy*phi_x**2) / (phi_x**2 + phi_y**2)
    abs_grad_phi_sq = phi_x**2 + phi_y**2
    
    # 避免除零
    epsilon = np.finfo(phi.dtype).eps
    
    numerator = (phi_xx * phi_y**2 - 
                 2 * phi_y * phi_x * phi_xy + 
                 phi_yy * phi_x**2)
    
    # 分母为0时，曲率也为0
    kappa_abs_phi = np.divide(numerator, abs_grad_phi_sq, 
                              out=np.zeros_like(phi), 
                              where=(abs_grad_phi_sq > epsilon))
    
    # 4. 乘以曲率系数
    delta = b * kappa_abs_phi
    
    return delta