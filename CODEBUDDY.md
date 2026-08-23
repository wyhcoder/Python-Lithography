# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## 项目概述

计算光刻仿真与 OPC 优化代码库，用于半导体制造中的光刻工艺仿真。包含两套独立的成像模型：

- **标量模型** (numpy/scipy, CPU)：基于 Hopkins 部分相干成像 + SOCS 加速，位于 `litho_model/` 和 `op_model/`
- **矢量模型** (PyTorch, CPU/GPU)：基于 Mansuripur 偏振矢量成像，位于 `projects/vector_imaging/`
- **独立 SOCS 成像系统** (numpy + PyTorch 双后端)：自包含的标量 SOCS 成像库，位于 `scalar_socs_imaging/`

高 NA (NA=1.35) 浸没式光刻下必须使用矢量模型，标量结果会系统性偏低约 20%。

## 环境配置

```bash
# Apple Silicon macOS
conda env create -f environments/macos.yml

# Windows 完整/最小历史环境
conda env create -f environments/windows-full.yml
conda env create -f environments/windows-min.yml

conda activate litho
```

## 运行命令

### 标量模型案例脚本 (项目根目录运行)

```bash
python -m examples.scalar.case_1                   # 完整流程：CTM → Level-Set 优化
python -m examples.scalar.case_2                   # PV-Band 诊断
python -m examples.scalar.case_meef                # MEEF 曲线优化
python -m examples.scalar.case_meef_autograd       # 自动微分版 MEEF
python -m examples.scalar.case_meef_pipeline       # 独立 MEEF Pipeline
python -m examples.scalar.case_sraf                # SRAF 宽度/位置优化
```

### 矢量模型 (projects/vector_imaging/ 目录下运行)

```bash
cd projects/vector_imaging

# 端到端 demo + autograd 验证
python forward_demo.py --type vector --method socs --K 20

# 批量 MEEF 优化
python meef_batch_demo.py --iterations 30 --delta 0.5 --batch 64

# 批量仿真
python batch_simulate.py --type vector --batch 32
```

### 测试/诊断脚本 (项目根目录运行)

```bash
python -m tests.test_batch_fft_consistency    # FFT 一致性验证
python -m tests.test_meef_pipeline_setup      # Pipeline 装配测试
python -m tests.test_sdf_vs_msaa              # SDF vs MSAA 渲染对比
python -m tests.verify_meef_autograd_vs_fd    # Autograd vs 有限差分验证
```

## 高层架构

### 目录结构

```
├── litho_model/         # 标量光刻仿真核心 (numpy)
│   ├── lithography_simulator.py   # 顶层模拟器，组装所有组件
│   ├── simulation_parameters.py   # 参数数据类 + from_yaml() 工厂方法
│   ├── config.yaml                # 默认仿真参数 (193nm, NA=1.35, 环形光源)
│   ├── set_patterns.py            # Mask 类
│   ├── set_optical_system.py      # OpticalSystem 类 (Zernike 像差)
│   ├── set_illumination.py        # Illumination 类 (光源)
│   ├── set_meef.py                # MEEF 类 (控制点/评估点)
│   ├── set_sraf.py                # SRAF 类 (分阶/骨架化/B-spline)
│   ├── demo_compute_image.py      # 前向仿真 (v1/v2/v3 三代实现)
│   └── load_mask_bmp.py           # BMP 版图加载
│
├── op_model/           # 优化算法库 (numpy)
│   ├── demo_pe_ctm.py             # CTM 优化器
│   ├── demo_pe_gradient.py        # PE 解析梯度计算
│   ├── demo_level_set.py          # Level-Set 优化器
│   ├── demo_level_set_methods/    # WENO/FMM/Godunov 数值方法
│   ├── demo_meef.py               # MEEF 优化器 (中心差分 + TSVD)
│   ├── demo_meef_autograd/        # Autograd 版 MEEF (L-BFGS / Gauss-Newton)
│   ├── meef_pipeline/             # 独立 Pipeline (LSM→CP→B-spline→MEEF)
│   ├── demo_pv_band.py            # PV-Band 计算
│   ├── demo_pv_gradient.py        # PV 梯度
│   └── demo_sraf.py               # SRAF 宽度/位置优化器
│
├── utils_model/        # 工具函数
│   ├── select_eps.py              # CP/EP 选取
│   ├── project_paths.py           # 项目路径唯一来源
│   ├── demo_MSAA.py               # MSAA 抗锯齿渲染器
│   ├── demo_parametric.py         # B-spline 参数化曲线
│   ├── demo_sdf_renderer.py       # SDF + sigmoid 渲染器
│   └── plot_matrix.py             # 可视化
│
├── projects/vector_imaging/  # 矢量光刻模型 (PyTorch)
│   ├── configs/litho_system.yaml  # 矢量模型配置
│   ├── litho_model/              # 模型核心
│   │   ├── config.py             # SimulationConfig 数据类
│   │   ├── socs.py               # VectorSOCS / ScalarSOCS 分解
│   │   ├── forwardImage.py       # 前向成像 (VectorForwardImaging 等)
│   │   ├── pupil.py              # 复光瞳函数
│   │   ├── source.py             # 光源 + SourceType 枚举
│   │   ├── vector_transfer.py    # Mansuripur 偏振传递矩阵 (3x2)
│   │   ├── mask.py               # MaskDataset (DataLoader 适配)
│   │   └── meef_opt/             # 批量 MEEF 优化器
│   ├── forward_demo.py           # 端到端 demo
│   ├── batch_simulate.py         # 批量仿真
│   └── meef_batch_demo.py        # 批量 MEEF demo
│
├── examples/scalar/     # 标量/OPC 案例驱动脚本
├── scalar_socs_imaging/ # 独立标量 SOCS 包
├── data/patterns/       # 输入 mask 版图 BMP 文件
├── tests/               # 数值检查与烟雾测试
├── scripts/analysis/    # 绘图、诊断与转换工具
├── outputs/opc/         # 优化结果输出目录
└── docs/                # 框架说明与论文材料
```

### 核心仿真链路

```
config.yaml
    │
    ▼
SimulationParameters.from_yaml()   # 加载参数
    │
    ▼
LithographySimulator(params)       # 创建仿真器
    ├── Mask            (set_patterns.py)       ← data/patterns/*.bmp
    ├── OpticalSystem   (set_optical_system.py)  # Zernike + 切趾
    ├── Illumination    (set_illumination.py)    # 光源生成
    └── MEEF            (set_meef.py)            # CP/EP 选取
    │
    ▼
simulator.prepare_for_optimization(method="SOCS")  # 预计算 PSF/SOCS 核
    │
    ▼
images_simulation_v2(mask, resist_params, opt_cache)  # 前向仿真
    → (aerial_image, wafer_image, intermediates)
```

### 成像方法对比

| 特征 | ScalarForwardImaging | VectorForwardImaging | ScalarSOCSImaging | VectorSOCSImaging |
|------|---------------------|---------------------|-------------------|-------------------|
| 电场 | 单分量 | 三分量 (Ex, Ey, Ez) | 单分量 | 三分量 |
| 加速 | 无 (Abbe 逐点) | 无 (Abbe 逐点) | SVD 相干核 | SVD 相干核 |
| 框架 | numpy / torch | torch | numpy / torch | torch |
| 自动微分 | torch 版支持 | 支持 | torch 版支持 | 支持 |

### 前向仿真三代实现 (`demo_compute_image.py`)

- **v1**：逐核循环 `fftconvolve` — 最慢，仅作参考
- **v2**：逐核 FFT 卷积 — 当前稳定版，`prepare_for_optimization()` 预存各核 FFT
- **v3**：批量堆叠 FFT — `H_k_fft_stack` 广播一次完成所有核，约 3-5x 加速

## 关键概念

- **SOCS**: Sum of Coherent Systems，SVD 分解 Hopkins 互强度函数，取前 K 个相干核 (K≈50)
- **CTM**: Cosine Transform Method，余弦变换优化
- **MEEF**: Mask Error Enhancement Factor，掩模误差增强因子
- **EPE/wEPE**: Edge Placement Error / weighted EPE，边缘放置误差
- **PE**: Pattern Error，版图误差
- **PV-Band**: Process Variation Band，工艺变化带
- **CP/EP**: Control Point / Evaluation Point，控制点/评估点
- **SRAF**: Sub-Resolution Assist Feature，亚分辨率辅助图形
- **LSM**: Level-Set Method，水平集方法
- **SDF**: Signed Distance Function，符号距离函数

## 重要注意事项

- 标量模型和矢量模型共享光学参数，但实现路径独立，代码不可混用
- `environments/windows-full.yml` 与 `environments/windows-min.yml` 含 Windows 平台锁定依赖，macOS 不要使用
- 运行矢量模型前必须在当前环境验证 `import torch`，不能只依据环境文件判断
- `prepare_for_optimization()` 预计算 SOCS 核和 FFT 缓存，这些缓存必须以 `np.complex64` 格式存储在 `LithographySimulator.opt_cache` 中
- 数值检查集中在 `tests/`，人工绘图/诊断脚本集中在 `scripts/analysis/`
- 输入路径和输出路径统一从 `utils_model/project_paths.py` 解析
- MEEF 优化中 `delta` 为中心差分扰动量（像素），越大梯度估计越粗糙；`batch_size=0` 表示一次性全部仿真
