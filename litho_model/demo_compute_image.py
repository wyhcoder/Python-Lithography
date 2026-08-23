# import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
# # ----------------- 导入我们的仿真框架组件 -----------------
# from simulation_parameters import SimulationParameters
# from set_patterns import Mask
# from set_optical_system import OpticalSystem
# from set_illumination import Illumination 

# # ----------------- 核心：矢量化的Abbe成像函数 -----------------

# def compute_aerial_image(mask: Mask, optics: OpticalSystem, source: Illumination) -> np.ndarray:
#     """
#     使用矢量化（批处理）方法高效计算空间像。
#     此算法在数学上等同于原始的 for 循环 Abbe 模型。
#     """
#     print("\n--- Starting Vectorized Aerial Image Computation ---")
    
#     # 1. 准备输入数据
#     mask_spectrum = mask.spectrum
#     pupil_func = optics.pupil_function
#     source_map = source.source_map
    
#     # 2. 离散化光源
#     sindy, sindx = np.nonzero(source_map > 1e-9)
#     source_weights = source_map[sindy, sindx]
    
#     if len(sindx) == 0:
#         print("警告：光源中没有有效的发光点。返回全零图像。")
#         return np.zeros_like(mask.data)

#     print(f"Discretized source into {len(sindx)} points.")

#     # 3. 计算像素偏移量
#     pupil_k_step = optics.frequency_coords[1] - optics.frequency_coords[0]
#     source_k_coords_x = source.source_coords[sindx]
#     source_k_coords_y = source.source_coords[sindy]
#     shifts_x = np.round(source_k_coords_x / pupil_k_step).astype(int)
#     shifts_y = np.round(source_k_coords_y / pupil_k_step).astype(int)
    
#     # 4. 创建移位光瞳的批次
#     shifted_pupils_batch = np.array([
#         np.roll(pupil_func, (sy, sx), axis=(0, 1))
#         for sy, sx in zip(shifts_y, shifts_x)
#     ])
    
#     # 5. 批次滤波
#     filtered_spectra_batch = mask_spectrum[np.newaxis, :, :] * shifted_pupils_batch
    
#     # 6. 批次傅里叶逆变换
#     #    a. 将批次频谱从“中心”移到“角落”，以符合ifft2的要求
#     spectra_for_ifft = np.fft.ifftshift(filtered_spectra_batch, axes=(-2, -1))
    
#     #    b. 执行逆变换，得到原点在“角落”的空间电场批次
#     corner_origin_E_fields = np.fft.ifft2(spectra_for_ifft, axes=(-2, -1))
    
#     #    c. 将空间电场批次从“角落”移回“中心”，得到物理上正确的空间像
#     E_fields_batch = np.fft.fftshift(corner_origin_E_fields, axes=(-2, -1))
    
#     # 7. 计算强度并进行非相干叠加
#     intensities_batch = np.abs(E_fields_batch)**2
#     aerial_image = np.sum(intensities_batch * source_weights[:, np.newaxis, np.newaxis], axis=0)
    
#     # 8. 归一化
#     total_source_intensity = np.sum(source_weights)
#     if total_source_intensity > 1e-9:
#         aerial_image /= total_source_intensity
        
#     print("--- Computation Finished ---")
#     return aerial_image


# def compute_wafer_image(params: SimulationParameters, aerial_image: np.ndarray) -> np.ndarray:
#         """
#         使用 Sigmoid 模型计算晶圆像。
#         这是一个独立的辅助函数。
        
#         Args:
#             params (SimulationParameters): 包含所有仿真配置的参数对象。
#             aerial_image (np.ndarray): 要处理的空间像。

#         Returns:
#             np.ndarray: 计算得到的晶圆像。
#         """ 
#         alpha = params.resist.alpha
#         threshold = params.resist.threshold
        
#         wafer_image = 1 / (1 + np.exp(-alpha * (aerial_image - threshold)))
#         return wafer_image


import numpy as np
from .set_patterns import Mask
from .set_optical_system import OpticalSystem
from .set_illumination import Illumination
from .simulation_parameters import ResistParams
from typing import Tuple, List
import math
from utils_model.project_paths import DIAGNOSTICS_OUTPUT_DIR
def show_images(images, titles=None, ncols=3, cmap='gray'):
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
    # np.savetxt("mask_fft_python.txt", Asp)
    # np.savetxt("PSF_fft_python.txt", Bsp)
    # 在频域中进行点乘
    c = Asp * Bsp
    
    # 进行IFFT和ifftshift操作
    c = np.fft.ifftshift(np.fft.ifft2(c))
    
    return c 

def _compute_wafer_image(aerial_image: np.ndarray, resist_params: ResistParams) -> np.ndarray:
    """【计算核心】使用 Sigmoid 模型计算晶圆像。"""
    # print("Calculating Wafer Image...")
    alpha = resist_params.alpha
    threshold = resist_params.threshold
    wafer_image = 1 / (1 + np.exp(-alpha * (aerial_image - threshold)))
    return wafer_image


def images_simulation_v1(mask_spatial: np.ndarray, optics: OpticalSystem, 
                        source: Illumination, resist_params: ResistParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    快速的正向仿真，只返回最终结果，不缓存中间变量。
    使用完全矢量化的方法以获得最高性能。
    """
    print("\n--- Running Fast Forward Simulation (Vectorized) ---")
    
    # 1. 计算频谱
    mask_spectrum = Mask.calculate_spectrum_from_data(mask_spatial)
    
    # 2. 矢量化计算空间像
    pupil_func = optics.pupil_function
    source_map = source.source_map
    
    sindy, sindx = np.nonzero(source_map > 1e-9)
    source_weights = source_map[sindy, sindx]
    
    if len(sindx) == 0:
        zeros = np.zeros_like(mask_spatial, dtype=float)
        return zeros, zeros

    pupil_k_step = optics.frequency_coords[1] - optics.frequency_coords[0]
    source_k_coords_x = source.source_coords[sindx]
    source_k_coords_y = source.source_coords[sindy]
    shifts_x = np.round(source_k_coords_x / pupil_k_step).astype(int)
    shifts_y = np.round(source_k_coords_y / pupil_k_step).astype(int)
    
    shifted_pupils_batch = np.array([np.roll(pupil_func, (sy, sx), axis=(0, 1)) for sy, sx in zip(shifts_y, shifts_x)])
    filtered_spectra_batch = mask_spectrum[np.newaxis, :, :] * shifted_pupils_batch
    
    spectra_for_ifft = np.fft.ifftshift(filtered_spectra_batch, axes=(-2, -1))
    corner_origin_E_fields = np.fft.ifft2(spectra_for_ifft, axes=(-2, -1))
    E_fields_batch = np.fft.fftshift(corner_origin_E_fields, axes=(-2, -1))
    
    intensities_batch = np.abs(E_fields_batch)**2
    aerial_image = np.sum(intensities_batch * source_weights[:, np.newaxis, np.newaxis], axis=0)
    
    total_source_intensity = np.sum(source_weights)
    if total_source_intensity > 1e-9:
        aerial_image /= total_source_intensity
        
    wafer_image = _compute_wafer_image(aerial_image, resist_params)
    
    return aerial_image, wafer_image


def images_simulation_v2(mask_spatial: np.ndarray, 
                                resist_params: ResistParams, 
                                opt_cache: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    执行为梯度计算优化的正向仿真，利用预计算的缓存。
    返回最终结果以及梯度计算所需的、本次迭代的中间变量。
    """
    # print("--- Running Forward Simulation (for Gradient) using cached constants ---")
    
    # 1. 从缓存中获取预计算好的 H_k 列表和光源权重
    H_k_list = opt_cache["H_k_list"]
    source_weights = opt_cache["source_weights"] #是一个一维的数组，保存了有多少大于0的光源点的强度值
    #是否用SOCS方法
    if "norm_factor" in opt_cache:
        norm_factor = opt_cache["norm_factor"]
        socs = True

        if not opt_cache.get("_socs_printed", False):
            print("########### Using SOCS 成像 ############")
            opt_cache["_socs_printed"] = True
    else:
        socs = False

        if not opt_cache.get("_abbe_printed", False):
            print("########### Using Abbe ############")
            opt_cache["_abbe_printed"] = True
    # 2. 初始化本次迭代的中间变量
    iteration_intermediates = { "E_k_list": [] }
    aerial_image_acc = np.zeros_like(mask_spatial, dtype=np.float64)
    total_intensity = np.sum(source_weights)
    
    # 3. 循环计算与掩模相关的变量
    for i in range(len(source_weights)):
        H_k = H_k_list[i]
        weight = source_weights[i]
        
        # 循环内只剩下与 mask 相关的卷积运算
        E_k = fftconvolve1(mask_spatial, H_k)
        # if i == 0:
        #     # print("E_k shape: ", E_k.shape)
        #     np.savetxt(DIAGNOSTICS_OUTPUT_DIR / "scalar_text" / "E_k_python.txt", E_k)
        #     mask_spatial_fft =np.fft.fft2(mask_spatial)
        #     np.savetxt(DIAGNOSTICS_OUTPUT_DIR / "scalar_text" / "mask_spatial_fft_python.txt", mask_spatial_fft)
        #     H_k_fft = np.fft.fft2(H_k)
        #     np.savetxt(DIAGNOSTICS_OUTPUT_DIR / "scalar_text" / "PSF_fft_python.txt", H_k_fft)
        # E_k = fftconvolve(mask_spatial, H_k, mode="same")
        # np.savetxt(f"./TXT_test/E/E{i}.txt",E_k)
        iteration_intermediates["E_k_list"].append(E_k)
        
        aerial_image_acc += weight * np.abs(E_k)**2
    
    if socs:
        
        aerial_image = aerial_image_acc/norm_factor
        
        wafer_image = _compute_wafer_image(aerial_image, resist_params)
        # wafer_image = np.zeros_like(aerial_image)
        # wafer_image[aerial_image > 0.15] = 1
        images = [
        aerial_image_acc,
        aerial_image,
        wafer_image
        ]

        titles = [
            "Aerial Image (Accumulated) in socs",
            "Aerial Image (Normalized) in socs",
            "Wafer Image in cocs"
        ]

        # show_images(images, titles, ncols=3)
    else:
        # 4. 归一化空间像并计算晶圆像
        total_intensity = np.sum(source_weights)
        if total_intensity > 1e-9:
            aerial_image = aerial_image_acc / total_intensity
            
        else:
            aerial_image = aerial_image_acc
        
        print("total intensity: ", total_intensity)
        wafer_image = _compute_wafer_image(aerial_image, resist_params)
    
    # print("--- Forward run for gradient finished ---")
    return aerial_image, wafer_image, iteration_intermediates




def images_simulation_v3(mask_spatial: np.ndarray,
                                resist_params: ResistParams,
                                opt_cache: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Thread-safe forward simulation for gradient/MEEF calculation.
    
    优先走 batch FFT 路径：当 opt_cache 中存在预计算好的 H_k_fft_stack 时，
    一次性对所有 N 个 kernel 做卷积，避免 N 次 Python for 循环 + 单次 FFT。
    数学上与原本 fftconvolve1 的逐核卷积完全等价。

    缓存缺失时自动 fallback 到逐核循环，保证向后兼容。

    线程安全要点：
    - opt_cache 全程只读；
    - NumPy 的 fft2/ifft2 在 batch 维上是无状态的、且会释放 GIL，
      因此可以被多个线程同时调用而无需加锁。
    """
    H_k_fft_stack = opt_cache.get("H_k_fft_stack", None)
    source_weights = opt_cache["source_weights"]
    socs = "norm_factor" in opt_cache
    norm_factor = opt_cache["norm_factor"] if socs else None

    # 用 dict 而不是字面量约束类型，方便 fast/fallback 两条路径返回不同形态
    iteration_intermediates: dict = {}

    if H_k_fft_stack is not None:
        # ===== 快速路径：batch FFT =====
        # 与 fftconvolve1 等价: ifftshift(ifft2(fft2(mask) * fft2(H_k)))
        mask_fft = np.fft.fft2(mask_spatial)                       # (H, W) complex
        # broadcast 到 (N, H, W); 复数乘法在 BLAS 层是多线程的(若 numpy 启用了 MKL/OpenBLAS)
        spectra_stack = mask_fft[None, :, :] * H_k_fft_stack       # (N, H, W)
        E_stack = np.fft.ifftshift(
            np.fft.ifft2(spectra_stack, axes=(-2, -1)),
            axes=(-2, -1),
        )                                                          # (N, H, W) complex

        # |E|^2 在 batch 维与 source_weights 做加权求和: einsum 在 N 较小时也很高效
        intensities = (E_stack.real ** 2) + (E_stack.imag ** 2)    # (N, H, W) float
        aerial_image_acc = np.einsum('n,nhw->hw', source_weights, intensities)

        # 仅在下游真的需要 E_k_list 时才会去 list 化；MEEF 流程不会用到，
        # 这里仍然返回 stack 视图，避免不必要的拷贝
        iteration_intermediates["E_k_list"] = E_stack
    else:
        # ===== Fallback：与老版本完全一致 =====
        H_k_list = opt_cache["H_k_list"]
        aerial_image_acc = np.zeros_like(mask_spatial, dtype=np.float64)
        E_k_list_fallback = []
        for H_k, weight in zip(H_k_list, source_weights):
            E_k = fftconvolve1(mask_spatial, H_k)
            E_k_list_fallback.append(E_k)
            aerial_image_acc += weight * np.abs(E_k) ** 2
        iteration_intermediates["E_k_list"] = E_k_list_fallback

    if socs:
        aerial_image = aerial_image_acc / norm_factor
    else:
        total_intensity = np.sum(source_weights)
        if total_intensity > 1e-9:
            aerial_image = aerial_image_acc / total_intensity
        else:
            aerial_image = aerial_image_acc

    wafer_image = _compute_wafer_image(aerial_image, resist_params)
    return aerial_image, wafer_image, iteration_intermediates
