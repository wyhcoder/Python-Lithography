"""LSM mask 加载与主图形/SRAF 拆分.

输入:
    - LSM mask 的文件路径 (txt, 由 yaml 中 ls_mask_path 提供)
    - target_mask (二值, 仅含主图形)
输出:
    - lsm_mask:  原始 LSM 灰度 / 二值版图
    - main_mask: 主图形所在区域 (在 target_mask 膨胀范围内的 LSM 像素值)
    - sraf_mask_raw: 余下的 SRAF 区域 (LSM 中扣掉主图形)
"""

from pathlib import Path
import numpy as np
from numpy.ma import masked_not_equal
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

from utils_model.project_paths import project_path


def load_lsm_mask(ls_mask_path: str) -> np.ndarray:
    """从 yaml 配置的相对/绝对路径读取 LSM mask 的 txt.

    txt 每行用空格分隔, 数值范围一般在 [0, 1].
    """
    p = project_path(ls_mask_path)
    if not p.exists():
        raise FileNotFoundError(f"LSM mask 文件不存在: {p}")
    return np.loadtxt(p)


def align_to_target_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """把任意 2D 数组对齐到 target_shape: 大了就 center-crop, 小了就 zero-pad.

    项目里历史 LSM txt 多为 257x257, 而 target_mask 是 256x256, 这里
    自动差一像素裁切, 避免每次手动改数据.
    """
    arr = np.asarray(arr)
    H_t, W_t = target_shape
    H_s, W_s = arr.shape
    if (H_s, W_s) == (H_t, W_t):
        return arr

    # 1) center-crop (各方向单独处理, 大了裁, 小了 pad)
    out = arr
    # 高度方向
    if H_s > H_t:
        s = (H_s - H_t) // 2
        out = out[s:s + H_t, :]
    elif H_s < H_t:
        pad_top = (H_t - H_s) // 2
        pad_bot = H_t - H_s - pad_top
        out = np.pad(out, ((pad_top, pad_bot), (0, 0)), mode="constant")
    # 宽度方向
    H_now, W_now = out.shape
    if W_now > W_t:
        s = (W_now - W_t) // 2
        out = out[:, s:s + W_t]
    elif W_now < W_t:
        pad_left = (W_t - W_now) // 2
        pad_right = W_t - W_now - pad_left
        out = np.pad(out, ((0, 0), (pad_left, pad_right)), mode="constant")
    return out


def split_main_and_sraf(
    lsm_mask: np.ndarray,
    target_mask: np.ndarray,
    dilate_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """按 target_mask 膨胀区域拆分 LSM 为 main_mask 与 sraf_mask_raw.

    与 utils_model/demo_sraf_utils.extract_sraf 行为一致:
        - 不改变像素灰度, 仅按空间区域归类
        - 主图形允许区域 = 二值化(target_mask) 膨胀 dilate_radius 像素
    """
    full = np.asarray(lsm_mask, dtype=np.float32)
    origin_bin = (np.asarray(target_mask) > 0).astype(np.uint8)
    dilated = binary_dilation(origin_bin, structure=disk(dilate_radius))

    main_mask = np.zeros_like(full, dtype=full.dtype)
    main_mask[dilated] = full[dilated]

    sraf_raw = np.zeros_like(full, dtype=full.dtype)
    sraf_raw[~dilated] = full[~dilated]
    return main_mask, sraf_raw

    

    
