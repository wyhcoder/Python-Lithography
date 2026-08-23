import numpy as np

def get_dt_normal_kappa(alpha: float, dx: float, dy: float, 
                   H1_abs: np.ndarray, H2_abs: np.ndarray, b: float) -> float:
    """
    根据CFL条件计算稳定的时间步长 dt。
    
    参数:
        alpha: 安全因子 (0 < alpha < 1)。
        dx, dy: 网格间距。
        H1_abs, H2_abs: 法向演化项在x,y方向的通量绝对值。
        b: 曲率系数。
        
    返回:
        dt: 保证数值稳定的时间步长。
    """
    if not (0 < alpha < 1):
        raise ValueError("安全因子 alpha 必须在 0 和 1 之间")

    # 法向速度项的贡献
    # 使用 np.max 获取全局最大值
    normal_term = np.max(H1_abs) / dx + np.max(H2_abs) / dy
    
    # 曲率项的贡献 (这是一个二阶项，与 1/dx^2 相关)
    kappa_term = b * (1/dx**2 + 1/dy**2) * 2 # 乘以2是一个常见的保守估计
    
    # 总分母，并处理其为0的特殊情况
    denominator = normal_term + kappa_term
    if denominator < 1e-9: # 如果速度场和曲率都接近0
        return 1e-3 # 返回一个合理的默认小步长

    dt = alpha / denominator
    return dt