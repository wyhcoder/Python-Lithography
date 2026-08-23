# Vector MEEF Batch Optimizer

基于矢量 SOCS 成像的 MEEF (Mask Error Enhancement Factor) 批量优化器。

## 核心思路

迁移自 CPU 版的 `examples/scalar/case_meef_pipeline.py`，关键改进是 **批量仿真**：

```
传统逐 mask 仿真 (N=2*num_cps 张):
  for j in range(N):
      simulate(mask_j) → EPE_j        # 串行 N 次 FFT 卷积

批量仿真 (本实现):
  batch = stack([mask_0, mask_1, ..., mask_{N-1}])  # [N, H, W]
  aerials = forward_batch(batch)                     # 一次广播完成所有 FFT
  EPE_batch = compute_epe(aerials)                   # 一次梯度计算 + 索引
```

利用 `VectorSOCSImaging.forward_batch` 的 `[B, K, C, N, N]` 广播能力，
一次 IFFT 同时处理所有扰动 mask。

## 文件结构

```
meef_opt/
├── __init__.py          # 包入口，导出 VectorMEEFOptimizer
├── optimizer.py         # 主类: 批量 CFD + Gauss-Newton + L-curve TSVD
├── utils.py             # EPE/wEPE 计算、TSVD 求解器、CP/EP 提取
├── sdf_renderer.py      # SDF + sigmoid 软渲染 (纯 numpy, 自包含)
└── README.md            # 本文档
```

## 快速开始

入口脚本 `meef_batch_demo.py` 在 `projects/vector_imaging/` 根目录:

```bash
# 默认: contact hole 测试图形, 10 轮迭代, 全批量仿真
python3 meef_batch_demo.py

# 自定义
python3 meef_batch_demo.py \
    --iterations 30 \
    --delta 0.5 \
    --batch 64 \
    --sdf-beta 0.25 \
    --cp-interval 8 \
    --output ./output/meef_opt
```

参数说明:
- `--iterations`: 优化迭代次数
- `--delta`: 中心差分扰动量 (像素)
- `--batch`: 单批仿真 mask 数 (0=一次性全部，内存允许时最快)
- `--sdf-beta`: SDF 软化宽度 (越小越锐利)
- `--cp-interval`: CP 采样间隔 (越大 CP 越少，优化越快但精度降低)

## 编程接口

```python
from litho_model.meef_opt import VectorMEEFOptimizer
from litho_model.forwardImage import VectorSOCSImaging
# ... 构建 imaging (见 meef_batch_demo.py 的 build_optical_system)

config = {
    "delta": 0.5,
    "iterations": 30,
    "sdf_beta": 0.25,
    "threshold": 0.25,
    "curve_type": "OA",          # "OA"=折线, "BS"=B-spline
    "output_dir": "./output/meef",
    "batch_size": 0,              # 0=全部一起仿真
}

optimizer = VectorMEEFOptimizer(imaging, target_mask, config)

# 方式 1: 自动从 target 提取 CP/EP
optimizer.setup_auto(interval=8)

# 方式 2: 手动指定
# optimizer.setup_contour(cps_list, eps_px, ep_weights)

best_cps, best_mask = optimizer.run()
```

## 性能特点

实测 (M-series CPU, N=257, 约 30 个 CP, 80 个 EP):

| 配置 | 单轮耗时 | 备注 |
|------|---------|------|
| `batch=0` (全量仿真) | ~3s | 最快，需要内存能放下整个 batch |
| `batch=8` (分批) | ~5s | 内存友好 |
| 等价的 numpy 串行版 | ~30s+ | 参考基线 |

加速比主要来源:
1. **批量 FFT 广播**: `[B, K, C, N, N]` 张量一次完成所有扰动的 SOCS 卷积
2. **多线程渲染**: ThreadPoolExecutor 并行 SDF 渲染
3. **GPU 友好**: 改 `--device cuda` 后 batch 越大加速越明显

## 与 CPU 版对比

| 方面 | CPU 版 (`examples/scalar/case_meef_pipeline.py`) | Vector 版 (本实现) |
|------|----------------------------------|--------------------|
| 光学模型 | 标量 SOCS | **矢量 SOCS** (Mansuripur 偏振) |
| 仿真粒度 | 逐 mask FFT 卷积 | **批量 FFT 广播** |
| 渲染器 | MSAA + SDF 切换 | SDF (软边可微) |
| 数据流 | numpy 全程 | numpy + torch 混合 |
| 优化算法 | Gauss-Newton + L-curve TSVD | 同 (复用 CPU 版逻辑) |
| SRAF 支持 | ✓ 完整 SRAF pipeline | ✗ (后续可补) |
| 对称性约束 | ✓ left-right/diagonal/center | ✗ (后续可补) |

## 后续可扩展

1. **SRAF pipeline**: 移植 `MEEFPipelineSetup` 到 Vector 代码
2. **对称性约束**: 添加 left-right/diagonal CP 提取
3. **autograd 模式**: 用 PyTorch jacfwd 替代 CFD (需要可微 SDF + B-spline)
4. **B-spline 优化**: 当前 BS 模式用 scipy，可改用 torch 实现
5. **GPU 优化**: 大 batch 在 GPU 上加速最显著
