from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Mask:
    """单张 mask 的加载与居中放置。"""

    def __init__(
        self,
        mask_dir: str,
        mask_name: str,
        mask_size: int,
        normalize: bool = True,
    ):
        self.mask_dir = Path(mask_dir)
        self.mask_name = mask_name
        self.mask_path = str(self.mask_dir / mask_name)
        self.mask_size = mask_size
        self.normalize = normalize
        self.mask_raw = self.load_mask()
        self.mask = self._create_pixel_mask()

    def load_mask(self) -> np.ndarray:
        img = Image.open(self.mask_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        if self.normalize:
            arr = arr / 255.0
        return arr

    def _create_pixel_mask(self) -> np.ndarray:
        """将原始 mask 居中放置到 mask_size × mask_size 的网格中。"""
        h_raw, w_raw = self.mask_raw.shape
        canvas = np.zeros((self.mask_size, self.mask_size), dtype=np.float32)
        sy = (self.mask_size - h_raw) // 2
        sx = (self.mask_size - w_raw) // 2
        ey = min(sy + h_raw, self.mask_size)
        ex = min(sx + w_raw, self.mask_size)
        canvas[sy:ey, sx:ex] = self.mask_raw[: ey - sy, : ex - sx]
        return canvas

    def get_mask(self) -> torch.Tensor:
        return torch.from_numpy(self.mask)


class MaskDataset(Dataset):
    """
    批量 mask 数据集：扫描一个目录下的所有 bmp/png/jpg 文件，
    每次 __getitem__ 返回 (mask_tensor [N,N], filename) 对。

    参数
    ----
    mask_dir  : 图像目录路径。
    mask_size : 仿真网格边长（像素），图像居中放置。
    normalize : 是否归一化到 [0, 1]。
    extensions: 扫描的文件后缀列表（不区分大小写）。
    file_list : 若指定，则只加载该列表中的文件名（相对于 mask_dir）；
                None 表示扫描全部。
    """

    EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(
        self,
        mask_dir: str | Path,
        mask_size: int,
        normalize: bool = True,
        extensions: Optional[set] = None,
        file_list: Optional[List[str]] = None,
    ):
        self.mask_dir = Path(mask_dir)
        self.mask_size = mask_size
        self.normalize = normalize
        self.extensions = extensions or self.EXTENSIONS

        if file_list is not None:
            self.files = [self.mask_dir / f for f in file_list]
        else:
            self.files = sorted(
                p
                for p in self.mask_dir.iterdir()
                if p.suffix.lower() in self.extensions
            )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"在 {self.mask_dir} 下没有找到任何图像文件"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        if self.normalize:
            arr = arr / 255.0

        # 居中放置
        h_raw, w_raw = arr.shape
        canvas = np.zeros((self.mask_size, self.mask_size), dtype=np.float32)
        sy = (self.mask_size - h_raw) // 2
        sx = (self.mask_size - w_raw) // 2
        ey = min(sy + h_raw, self.mask_size)
        ex = min(sx + w_raw, self.mask_size)
        canvas[sy:ey, sx:ex] = arr[: ey - sy, : ex - sx]

        tensor = torch.from_numpy(canvas)  # [N, N]
        return tensor, path.name
