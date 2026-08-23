import numpy as np
from litho_model.simulation_parameters import ResistParams
from scipy.signal import fftconvolve
def fftconvolve1(A, B):
    """
    使用FFT实现二维卷积，与MATLAB的fftconvolve函数行为完全一致
    
    Args:
        A: 第一个输入矩阵
        B: 第二个输入矩阵
        
    Returns:
        numpy.ndarray: 卷积结果
        
    Notes:
        该实现完全匹配MATLAB代码：
        Asp = fft2(A);
        Bsp = fft2(B);
        c = Asp.*Bsp;
        c = ifftshift(ifft2(c));
    """
    # 对输入矩阵进行二维FFT
    Asp = np.fft.fft2(A)
    Bsp = np.fft.fft2(B)
    
    # 在频域中进行点乘
    c = Asp * Bsp
    
    # 进行IFFT和ifftshift操作
    c = np.fft.ifftshift(np.fft.ifft2(c))
    
    return c 


def compute_mask_pe_gradient(
    wafer_image: np.ndarray, 
    target_mask: np.ndarray, 
    iteration_intermediates: dict,
    opt_cache: dict,
    resist_params: ResistParams
) -> np.ndarray:
    """
    计算代价函数 L = sum((WaferImage - TargetMask)^2) 关于掩模的梯度
    Args:
    wafer_image (np.ndarray): 当前的晶圆成像图。
    target_mask (np.ndarray): 目标图形。
    iteration_intermediates (dict): 从 run_simulation_for_gradient 返回的本次迭代的中间变量。
    opt_cache (dict): 从 simulator.prepare_for_optimization() 返回的预计算常量。
    resist_params (ResistParams): 光刻胶参数。
    """
    # print("\n--- Computing Mask Gradient using cached constants ---")
    
    pe_erro = np.sum((wafer_image - target_mask) ** 2)
    
    # 1. 从缓存中解包中间变量
    E_k_list = iteration_intermediates["E_k_list"]
    H_k_list = opt_cache["H_k_list"]
    source_weights = opt_cache["source_weights"]
    
    if not E_k_list or not H_k_list or source_weights.size == 0:
        raise ValueError("Intermediates or opt_cache dictionaries are missing required data.")

   
    # 如果是 SOCS，会从缓存拿 norm_factor；如果是原本的阿贝，会退而求和 source_weights
    norm_factor = opt_cache.get("norm_factor", np.sum(source_weights))


    # 2. 计算反向传播的初始误差信号
    alpha = resist_params.alpha
    temp = (wafer_image - target_mask) * wafer_image * (1 - wafer_image)

    # 3. 循环累加每个光源点的梯度贡献
    Gm = np.zeros_like(target_mask, dtype=np.float64)
    
    for i in range(len(E_k_list)):
        elec = E_k_list[i]
        H = H_k_list[i]
        weight = source_weights[i]
        
        # inter = weight * fftconvolve(np.conj(elec) * temp, np.rot90(H, 2))
        # Gm += 2 * alpha * np.real(inter)
       
        inter = weight *fftconvolve1(np.conj(elec) * temp, np.rot90(H, 2))
        Gm += 2 * alpha *  np.real(inter)
    

    # 4. 归一化 (改用自动判断的 norm_factor)
    if norm_factor > 1e-9:
        Gm /= norm_factor
        
    return Gm, pe_erro