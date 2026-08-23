"""
MSAA + 中心差分 + 梯度下降版 MEEF 优化器。

核心思路
--------
1. MSAA 渲染 (AntiAliasRenderer-16x, 与原版 demo_meef.py 一致)
2. 对每个 CP 的 (x, y) 做中心差分扰动, 复用 _build_meef_matrix_xy_parallel_threads
   得到雅可比 Mx, My (num_eps × num_cps)
3. 标量梯度 = (M · weights).sum(axis=0), 等价于直接对 wEPE 做中心差分
4. 梯度下降: cps -= lr · grad (Adam 或普通 GD)

不使用 SDF 渲染、PyTorch autograd、TSVD / L-curve。

文件结构
--------
- ``optimizer.py``  : 主优化器 ``MEEF_GradientDescentOptimizer`` + numpy 版 ``SimpleAdam``
- ``torch_litho.py``  : (保留, 不再使用)
- ``torch_render.py`` : (保留, 不再使用)

使用入口见 ``examples/scalar/case_meef_autograd.py``。
"""
