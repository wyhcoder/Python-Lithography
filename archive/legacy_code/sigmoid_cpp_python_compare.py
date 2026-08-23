import sys
import os

# 将当前脚本的父目录（即项目根目录）加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Union, Optional
from utils_model.plot_matrix import plot_matrix

def load_eigen_complex_txt(filename):
    # 正则表达式匹配格式为 (real,imag) 的数
    # \s* 匹配可能的空格, ([^,)]+) 匹配数字部分
    regex = r'\(\s*([^,)]+)\s*,\s*([^,)]+)\s*\)'
    
    # 将文件内容读入，并转换为复数类型
    # dtype=[('re', float), ('im', float)] 先读成两个浮点数
    data = np.fromregex(filename, regex, dtype=[('re', float), ('im', float)])
    
    # 组合成真实的复数矩阵
    return data['re'] + 1j * data['im']
def show_matrices(
    matrices: Union[np.ndarray, List[np.ndarray]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Optional[tuple] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,   # 新增：保存路径
    dpi: int = 600                   # 新增：保存分辨率
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
        im = ax.imshow(np.abs(mat), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(f"{save_path}/{filename}.png", dpi=dpi, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    plt.show()

def _compute_wafer_image(aerial_image: np.ndarray) -> np.ndarray:

    """【计算核心】使用 Sigmoid 模型计算晶圆像。"""
    # print("Calculating Wafer Image...")
    alpha =85
    threshold = 0.25
    wafer_image = 1 / (1 + np.exp(-alpha * (aerial_image - threshold)))
    return wafer_image

PSF_cpp = np.loadtxt(r'D:\Litho_Modle_Cpp\PSF_cpp.txt',dtype=complex)
PSF_python = np.loadtxt(r'E:\桌面\Litho_Model_cpu_v1.0\PSF.txt', dtype=complex)
mask_cpp = np.loadtxt(r'D:\Litho_Modle_Cpp\mask_cpp.txt', dtype=complex)
mask_fft_python = np.loadtxt(r'E:\桌面\Litho_Model_cpu_v1.0\mask_spatial_fft_python.txt', dtype=complex)
PSF_fft_python = np.loadtxt(r'E:\桌面\Litho_Model_cpu_v1.0\PSF_fft_python.txt', dtype=complex)
mask_fft_cpp = np.loadtxt(r'D:\Litho_Modle_Cpp\mask_fft_cpp.txt', dtype=complex)
PSF_fft_cpp = np.loadtxt(r'D:\Litho_Modle_Cpp\PSF_fft_cpp.txt', dtype=complex)
# plot_matrix(PSF_cpp, 'PSF_cpp')
# plot_matrix(PSF_python, 'PSF_python')
# plot_matrix(mask_cpp, 'mask_cpp')
plot_matrix(mask_fft_python, 'mask_fft_python')
plot_matrix(mask_fft_cpp, 'mask_fft_cpp')
plot_matrix(PSF_fft_cpp, 'PSF_fft_cpp')
plot_matrix(PSF_fft_python, 'PSF_fft_python')

# print(np.sum(PSF_cpp - PSF_python))
# E_K_python = np.loadtxt(r'E:\桌面\Litho_Model_cpu_v1.0\E_k_python.txt',dtype=complex)
# E_K_cpp = np.loadtxt(r'D:\Litho_Modle_Cpp\E_k_cpp.txt',dtype=complex)
# print(np.sum(E_K_python - E_K_cpp))
# plot_matrix(E_K_python, 'E_k_python ')
# plot_matrix(E_K_cpp, 'E_k_cpp')


# ai = np.loadtxt(r'D:\Litho_Modle_Cpp\wafer_image.txt')
# plt.imshow(ai)
# plt.show()
# wafer_image = _compute_wafer_image(ai)
# plt.imshow(wafer_image)
# plt.show()
# self_pupil = np.loadtxt(r'.\pupil_function.txt',dtype=complex)
# plot_matrix(self_pupil, 'self_pupil')

# self_kernel_spatial = np.loadtxt(r'.\kernel_spatial_0.txt',dtype=complex)
# plot_matrix(self_kernel_spatial, 'self_kernel_spatial')

# kernel_spatial = np.loadtxt(r'D:\Litho_Modle_Cpp\kernel_spatial.txt',dtype=complex)

# cppsource = np.loadtxt(r'D:\Litho_Modle_Cpp\source_map.txt')
# python_source = np.loadtxt(r'E:\桌面\Litho_Model_cpu_v1.0\self.source_map.txt')
# show_matrices([cppsource,python_source], ['cppsource','python_source'])
