import numpy as np
from matplotlib import pyplot as plt
def der_weno5(data: np.ndarray, dx: float, direction: str, axis: int = -1) -> np.ndarray:
    """
    使用五阶WENO格式计算N维数据
    """
    if direction not in ['plus', 'minus']:
        raise ValueError("Direction must be 'plus' or 'minus'")

    # --- 1. 修正边界处理  ---
    pad_width = [(0, 0)] * data.ndim
    if data.ndim > 0:
        pad_width[axis] = (3, 3)
    data_padded = np.pad(data, pad_width, mode='edge')

    # --- 2. 一阶差分  ---
    D1 = np.diff(data_padded, axis=axis) / dx
    
    # --- 3. 定义WENO5-JS系数  ---
    C = {
        '1/3': 1/3, '5/6': 5/6, '1/6': 1/6, '7/6': 7/6, '11/6': 11/6,
        'S1': 13/12, 'S2': 0.25
    }

    # --- 4. 切片 ---
    slices = [slice(None)] * data.ndim
    v_slices = []
    for i in range(5):
        current_slices = list(slices)
        current_slices[axis] = slice(i, i + data.shape[axis])
        v_slices.append(D1[tuple(current_slices)])
    
    if direction == 'minus':
        gamma1, gamma2, gamma3 = 0.1, 0.6, 0.3
        v1, v2, v3, v4, v5 = v_slices
    else: # 'plus'
        gamma1, gamma2, gamma3 = 0.3, 0.6, 0.1
        v5, v4, v3, v2, v1 = v_slices
        
    # --- 5. 核心计算  ---
    d1 = C['1/3']*v1 - C['7/6']*v2 + C['11/6']*v3
    d2 = -C['1/6']*v2 + C['5/6']*v3 + C['1/3']*v4
    d3 = C['1/3']*v3 + C['5/6']*v4 - C['1/6']*v5
    
    S1 = C['S1']*(v1 - 2*v2 + v3)**2 + C['S2']*(v1 - 4*v2 + 3*v3)**2
    S2 = C['S1']*(v2 - 2*v3 + v4)**2 + C['S2']*(v2 - v4)**2
    S3 = C['S1']*(v3 - 2*v4 + v5)**2 + C['S2']*(3*v3 - 4*v4 + v5)**2

    v_sq_list = [v**2 for v in v_slices]
    epsilon = 1e-6 * np.maximum.reduce(v_sq_list) + 1e-99
        
    alpha1 = gamma1 / (S1 + epsilon)**2
    alpha2 = gamma2 / (S2 + epsilon)**2
    alpha3 = gamma3 / (S3 + epsilon)**2
    
    w_total = alpha1 + alpha2 + alpha3
    w1 = alpha1 / w_total
    w2 = alpha2 / w_total
    w3 = alpha3 / w_total
    
    # --- 6. 加权组合 ---
    return w1 * d1 + w2 * d2 + w3 * d3

# 原代码

# def der_WENO_plus(data: np.ndarray, dx: float) -> np.ndarray:
#     """
#     使用五阶WENO格式计算一维数据的=正方向导数=（向量化版本）

#     """
#     data_x = np.zeros_like(data)
    
#     # 1. 外推边界点 (逻辑不变)
#     data = data.copy()
#     data[2] = 2*data[3] - data[4]
#     data[1] = 2*data[2] - data[3]
#     data[0] = 2*data[1] - data[2]
#     data[-3] = 2*data[-4] - data[-5]
#     data[-2] = 2*data[-3] - data[-4]
#     data[-1] = 2*data[-2] - data[-3]
    
#     # 2. 计算一阶差分 (逻辑不变)
#     D1 = np.diff(data) / dx
#     n = len(D1)
    
#     # 3. 准备数据 (v slicing is mirrored from minus-version, this is correct)
#     # v1 is right-most, v5 is left-most
#     v1 = D1[4:(n)]
#     v2 = D1[3:(n-1)]
#     v3 = D1[2:(n-2)]
#     v4 = D1[1:(n-3)]
#     v5 = D1[:(n-4)]
    
#     # --- 【关键修正区域 START】 ---

#     # 4. 计算三个候选导数 (完全镜像 der_WENO_minus 的公式)
#     #    minus d3(v3,v4,v5) -> plus d1(v3,v2,v1)
#     #    minus d2(v2,v3,v4) -> plus d2(v4,v3,v2)
#     #    minus d1(v1,v2,v3) -> plus d3(v5,v4,v3)
#     data_x_1 =  (1/3)*v3 + (5/6)*v2 - (1/6)*v1
#     data_x_2 = (-1/6)*v4 + (5/6)*v3 + (1/3)*v2
#     data_x_3 = (1/3)*v5 - (7/6)*v4 + (11/6)*v3
#     # 5. 计算平滑度指标 (同样完全镜像 der_WENO_minus 的公式)
#     #    minus S3(v3,v4,v5) -> plus S1(v3,v2,v1)
#     #    minus S2(v2,v3,v4) -> plus S2(v4,v3,v2)
#     #    minus S1(v1,v2,v3) -> plus S3(v5,v4,v3)
#     S1 = (13/12)*(v3-2*v2+v1)**2 + 0.25*(3*v3-4*v2+v1)**2
#     S2 = (13/12)*(v4-2*v3+v2)**2 + 0.25*(v4-v2)**2
#     S3 = (13/12)*(v5-2*v4+v3)**2 + 0.25*(v5-4*v4+3*v3)**2
    
#     epsilon = 1e-40
    
#     # 6. 计算权重 (理想权重系数 gamma 必须是镜像的: 0.3, 0.6, 0.1)
#     alpha1 = 0.3/((S1+epsilon)**2)
#     alpha2 = 0.6/((S2+epsilon)**2)
#     alpha3 = 0.1/((S3+epsilon)**2)
#     a_total = alpha1 + alpha2 + alpha3
    
#     # --- 【关键修正区域 END】 ---
    
#     # 7. 计算加权组合 (逻辑不变)
#     result_length = len(v1)
#     data_x[3:3+result_length] = (alpha1*data_x_1 + alpha2*data_x_2 + alpha3*data_x_3) / a_total
    
#     return data_x


# def der_WENO_minus(data: np.ndarray, dx: float) -> np.ndarray:
#     """
#     使用五阶WENO格式计算一维数据的=负方向导数=（向量化版本）
    
#     参数:
#         data: 输入数据（需要在开始和结束处扩展3个点）
#         dx: 网格分辨率
        
#     返回:
#         data_x: 计算得到的导数
#     """
#     # 创建输出数组
#     data_x = np.zeros_like(data)
    
#     # 外推边界点
#     data = data.copy()  # 避免修改输入数据
#     data[2] = 2*data[3] - data[4]
#     data[1] = 2*data[2] - data[3]
#     data[0] = 2*data[1] - data[2]
#     data[-3] = 2*data[-4] - data[-5]
#     data[-2] = 2*data[-3] - data[-4]
#     data[-1] = 2*data[-2] - data[-3]
    
#     # 计算一阶差分
#     D1 = np.diff(data) / dx
    
#     # 对于长度为n的输入，D1的长度为n-1
#     n = len(D1)
    
#     # 准备数据 - 确保所有切片长度相同
#     v1 = D1[:(n-4)]    # k+1
#     v2 = D1[1:(n-3)]   # k+2
#     v3 = D1[2:(n-2)]   # k+3
#     v4 = D1[3:(n-1)]   # k+4
#     v5 = D1[4:n]       # k+5
    
#     # 计算三个候选导数
#     data_x_1 = v1/3 - 7*v2/6 + 11*v3/6
#     data_x_2 = -v2/6 + 5*v3/6 + v4/3
#     data_x_3 = v3/3 + 5*v4/6 - v5/6
    
#     # 计算平滑度指标
#     epsilon = 1e-6 * np.maximum.reduce([v1**2, v2**2, v3**2, v4**2, v5**2]) + 1e-99
    
#     S1 = (13/12)*(v1-2*v2+v3)**2 + 0.25*(v1-4*v2+3*v3)**2
#     S2 = (13/12)*(v2-2*v3+v4)**2 + 0.25*(v2-v4)**2
#     S3 = (13/12)*(v3-2*v4+v5)**2 + 0.25*(3*v3-4*v4+v5)**2
    
#     # 计算权重
#     alpha1 = 0.1/((S1+epsilon)**2)
#     alpha2 = 0.6/((S2+epsilon)**2)
#     alpha3 = 0.3/((S3+epsilon)**2)
#     a_total = alpha1 + alpha2 + alpha3
    
#     # 计算加权组合
#     result_length = len(v1)  # 所有切片现在应该有相同的长度
#     data_x[3:3+result_length] = (alpha1*data_x_1 + alpha2*data_x_2 + alpha3*data_x_3) / a_total
    
#     return data_x
def main():
    # 包含突变的函数
    x = np.linspace(0, 1, 100)
    f = np.where(x < 0.5, x**2, 2*x - 0.25)  # 在 0.5 处连续但导数不连续

    # 简单 diff:
    f_diff = np.diff(f) / 0.01  # 在突变处剧烈震荡（Gibbs）

    # WENO5:
    f_weno = der_weno5(f, 0.01, 'plus', axis=0)  # 在突变处干净，无震荡
    


