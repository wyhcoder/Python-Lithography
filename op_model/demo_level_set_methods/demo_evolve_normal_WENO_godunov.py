# 文件名: evolve_normal_WENO_godunov.py
import numpy as np
from typing import Tuple
# 假设你已经更新了 der_weno5.py
from .demo_der_weno5 import der_weno5 

def select_upwind_deriv(Vn: np.ndarray, der_minus: np.ndarray, der_plus: np.ndarray) -> np.ndarray:
    """
    使用Godunov/Upwind格式选择正确的偏导数。
    Vn > 0 -> der_minus
    Vn < 0 -> der_plus
    Vn = 0 -> 导数为0 (或取平均，这里取0)
    """
    # 当Vn>0时，phi_der=der_minus；当Vn<0时，phi_der=der_plus
    phi_der = np.where(Vn > 0, der_minus, der_plus)
    # 当Vn=0时，演化停止，速度项为0
    phi_der = np.where(Vn == 0, 0, phi_der)
    return phi_der

def evolve_normal_WENO_godunov(phi: np.ndarray, dx: float, dy: float, Vn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用5阶WENO和Godunov通量计算法向演化量。
    """
    # 1. 扩展边界
    data_ext = np.pad(phi, 3, mode='edge')
    Vn_ext = np.pad(Vn, 3, mode='edge')

    # 2. 计算X方向导数 
    phi_x_minus = der_weno5(data_ext, dx, 'minus', axis=1)
    phi_x_plus = der_weno5(data_ext, dx, 'plus', axis=1)
    
    # 3. 计算Y方向导数 (使用转置技巧)
    phi_T = data_ext.T
    phi_y_minus_T = der_weno5(phi_T, dy, 'minus', axis=1)
    phi_y_plus_T = der_weno5(phi_T, dy, 'plus', axis=1)
    phi_y_minus = phi_y_minus_T.T
    phi_y_plus = phi_y_plus_T.T
    
    # 4. 使用Godunov/Upwind格式选择最终的 phi_x 和 phi_y
    phi_x = select_upwind_deriv(Vn_ext, phi_x_minus, phi_x_plus)
    phi_y = select_upwind_deriv(Vn_ext, phi_y_minus, phi_y_plus)
    
    # 5. 根据数学公式计算梯度范数
    grad_mag_sq = phi_x**2 + phi_y**2
    grad_mag = np.sqrt(grad_mag_sq)

    # 6. 计算最终演化项 delta = Vn * |∇φ|
    delta_ext = Vn_ext * grad_mag
    
    # 7. 为CFL条件计算max(|Vn*phi_x|), max(|Vn*phi_y|)
    H1_abs_ext = np.abs(Vn_ext * phi_x)
    H2_abs_ext = np.abs(Vn_ext * phi_y)

    # 8. 裁剪并返回
    delta = delta_ext[3:-3, 3:-3]
    H1_abs = H1_abs_ext[3:-3, 3:-3]
    H2_abs = H2_abs_ext[3:-3, 3:-3]
    
    return delta, H1_abs, H2_abs