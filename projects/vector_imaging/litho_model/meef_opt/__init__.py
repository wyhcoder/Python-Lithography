"""
MEEF 批量优化模块。

核心特性:
    - 利用 VectorSOCSImaging.forward_batch 一次性仿真所有扰动 mask
    - 中心差分 (CFD) 构建 MEEF 矩阵，多线程并行
    - Gauss-Newton + L-curve TSVD 求解 CP 位移
"""

from .optimizer import VectorMEEFOptimizer
