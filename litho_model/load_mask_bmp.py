import numpy as np
from PIL import Image

from utils_model.project_paths import PATTERN_DATA_DIR

def load_mask_image(image_name: str) -> np.ndarray:
    """读取Images文件夹下的掩模图像
    
    Args:
        image_name: 图像名称（如 'img9'，不需要.bmp后缀）
    
    Returns:
        np.ndarray: 掩模数据，归一化后的灰度图像
    """
    # 输入数据统一放在 data/patterns，不再依赖启动时的工作目录。
    image_path = PATTERN_DATA_DIR / f"{image_name}.bmp"
    
    # 读取图像并转换为numpy数组
    with Image.open(image_path) as img:
        # 先转换为数组，再转换类型
        mask_data = np.asarray(img).astype(np.float64)
    
    # 归一化处理：除以最大值
    mask_data = mask_data / np.max(mask_data)
    
    return mask_data 
