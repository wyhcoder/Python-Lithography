import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from PIL import Image
def save_mask_as_bmp(mask: np.ndarray, filename: str):
    """
    将 mask 数组保存为 BMP 灰度图

    Parameters
    ----------
    mask : np.ndarray
        输入矩阵（float / int，任意范围）
    filename : str
        输出文件名，例如 'mask.bmp'
    """

    # 归一化到 [0, 255]
    mask_norm = mask - mask.min()
    if mask_norm.max() > 0:
        mask_norm = mask_norm / mask_norm.max()

    mask_uint8 = (mask_norm * 255).astype(np.uint8)

    img = Image.fromarray(mask_uint8, mode='L')
    img.save(filename, format='BMP')
def plot_matrix(matrix: np.ndarray, title: str = "Matrix Visualization", cmap: str = 'viridis', 
                plot_type: str = 'amplitude', save_path: str = None) -> None:
    """高精度可视化矩阵
    
    Args:
        matrix: 要可视化的矩阵
        title: 图像标题
        cmap: 颜色映射，默认为'gray'
        plot_type: 对于复数矩阵，选择显示类型：
                  'amplitude' - 显示幅度（默认）
                  'phase' - 显示相位
                  'real' - 显示实部
                  'imag' - 显示虚部
        
    Example:
        >>> matrix = np.random.rand(100, 100)
        >>> plot_matrix(matrix, "Random Matrix")
        >>> complex_matrix = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
        >>> plot_matrix(complex_matrix, "Complex Matrix", plot_type='phase')
    """
    # 处理复数矩阵
    if np.iscomplexobj(matrix):
        if plot_type == 'amplitude':
            display_matrix = np.abs(matrix)
            title = f"{title} (Amplitude)"
        elif plot_type == 'phase':
            display_matrix = np.angle(matrix)
            title = f"{title} (Phase)"
        elif plot_type == 'real':
            display_matrix = np.real(matrix)
            title = f"{title} (Real Part)"
        elif plot_type == 'imag':
            display_matrix = np.imag(matrix)
            title = f"{title} (Imaginary Part)"
        else:
            raise ValueError("Invalid plot_type for complex matrix. Must be one of: 'amplitude', 'phase', 'real', 'imag'")
    else:
        display_matrix = matrix
        
    
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 显示矩阵
    im = plt.imshow(display_matrix, cmap=cmap, interpolation='nearest')
    
    # 添加颜色条
    plt.colorbar(im)
    
    # 设置标题（包含矩阵维度信息）
    plt.title(f"{title}\nShape: {matrix.shape}")
    
    # 设置轴标签
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # 添加网格
    plt.grid(False)
    
    # 显示图像
    plt.show() 
    
    # 新增保存功能
    if save_path is not None:
        # 保存时仅包含主图区域（移除所有周边空白和元素）
        plt.savefig(save_path, 
                   bbox_inches='tight',  # 自动裁剪空白
                   pad_inches=0,         # 填充边距设为0
                   dpi=300)
        
        
        
def saved_matrix(matrix, cmap='viridis', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 显示矩阵（关闭坐标轴和边框）
    im = ax.imshow(matrix, cmap)
    ax.axis('off')  # 关闭坐标轴
    
    # 不显示颜色条
    # plt.colorbar(im)  # 注释掉这行
    
    if save_path:
        plt.savefig(save_path, 
                   bbox_inches='tight', 
                   pad_inches=0,
                   dpi=300)
    
    plt.close()  # 关闭图形（非交互模式下必须）
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_psf(H_k, title="Point Spread Function (PSF)"):
    """
    可视化 PSF 的 2D 强度图和 3D 表面图
    :param H_k: 逆傅里叶变换后得到的复数矩阵 (PSF)
    """
    # 1. 计算强度 (Magnitude Squared)
    psf_intensity = np.abs(H_k)**2
    
    # 归一化，方便观察
    psf_intensity /= np.max(psf_intensity)

    # 准备坐标网格
    h, w = psf_intensity.shape
    x = np.linspace(-w//2, w//2, w)
    y = np.linspace(-h//2, h//2, h)
    X, Y = np.meshgrid(x, y)

    # 创建画布
    fig = plt.figure(figsize=(14, 6))

    # --- 2D 可视化 ---
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(psf_intensity, extent=[x[0], x[-1], y[0], y[-1]], cmap='magma')
    fig.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title(f"{title} - 2D Map")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")

    # --- 3D 可视化 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # rstride 和 cstride 控制采样步长，防止大矩阵绘图过慢
    surf = ax2.plot_surface(X, Y, psf_intensity, cmap='magma', 
                            linewidth=0, antialiased=True,
                            rstride=2, cstride=2)
    ax2.set_title(f"{title} - 3D Surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Intensity")
    
    # 调整视角
    ax2.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()
    
def show_matrices(
    matrices: Union[np.ndarray, List[np.ndarray]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    cmap: str = "gray",
    figsize: Optional[tuple] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
    show_titles: bool = False,
    show_colorbar: bool = False,
):
    """
    可视化一个或多个矩阵 (并排显示, 默认黑底白形, 像光罩示意).

    Parameters
    ----------
    matrices : np.ndarray or List[np.ndarray]
        单个矩阵或矩阵列表
    titles : List[str], optional
        每个矩阵对应的标题; 仅 show_titles=True 时显示
    save_path : str, optional
        保存目录 (None 则不保存)
    filename : str, optional
        保存文件名 (不含扩展名). 与 save_path 同时给出时才保存
    cmap : str
        颜色映射. 默认 "gray" 给出黑底白形; 也可设 "viridis"/"hot" 等
    figsize : tuple, optional
        图像尺寸. 不给时自动按矩阵数量等比放大
    vmin, vmax : float, optional
        颜色范围. 多矩阵对比时建议显式设置
    dpi : int
        保存分辨率
    show_titles : bool
        是否显示子图标题. 默认 False (出图最干净)
    show_colorbar : bool
        是否显示颜色条. 默认 False
    """

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
    if n == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        if show_titles:
            ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_linewidth(0.5)

    plt.tight_layout(pad=0.4)

    if save_path is not None and filename is not None:
        out = f"{save_path}/{filename}.png"
        plt.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"图像已保存至: {out}")
    plt.close(fig)


def plot_matrix_single(matrix, title, save_path=None, filename=None):
    plt.imshow(matrix, cmap = 'inferno', vmin=0, vmax=1)
    plt.title(title)
    if save_path is not None and filename is not None:
        out = f"{save_path}/{filename}.png"
        plt.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.05)
        print(f"图像已保存至: {out}")
    plt.close()