# Litho Simulation CPU

这是一个面向计算光刻仿真与 OPC 优化的 Python 工程，包含三条相对独立的成像链路：

```text
输入版图 data/patterns/*.bmp
          │
          ├── 旧标量主线 ── litho_model ──> op_model ──> outputs/opc
          │                    │                │
          │                    └── utils_model ─┘
          │
          ├── 独立标量 SOCS ── scalar_socs_imaging
          │
          └── 高 NA 矢量模型 ── projects/vector_imaging ──> 自己的 output
```

- `litho_model/`：旧标量光刻仿真核心，负责 mask、光源、光瞳、Abbe/SOCS 和光刻胶成像。
- `op_model/`：CTM、Level-Set、MEEF、PV-Band、SRAF 等优化算法。
- `utils_model/`：控制点、B-spline、MSAA、SDF、绘图及统一项目路径。
- `scalar_socs_imaging/`：自包含的标量 SOCS，实现 NumPy 与 PyTorch 两个后端。
- `projects/vector_imaging/`：独立的高 NA 偏振矢量成像子项目。

两处代码都使用了 `litho_model` 这个包名，因此矢量子项目必须从自己的目录启动，不能和根目录标量包混在同一个 Python 启动路径中。详细架构见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 1. 环境安装

项目建议使用 Python 3.10。Apple Silicon macOS 可直接创建项目环境：

```bash
git clone https://github.com/wyhcoder/Python-Lithography.git
cd Python-Lithography
conda env create -f environments/macos.yml
conda activate litho
```

如果本机已经存在 `litho` 环境，用环境文件补齐新增依赖：

```bash
conda env update -n litho -f environments/macos.yml
conda activate litho
```

验证关键依赖和实际解释器：

```bash
which python
python -c "import numpy, scipy, cv2, skimage, torch, yaml, cma; print('环境正常')"
```

`environments/windows-full.yml` 和 `environments/windows-min.yml` 是历史 Windows 环境，不要在 macOS 上使用。第一次导入 Matplotlib 时建立字体缓存属于正常现象。

## 2. 数据与配置

根目录标量 demo 共用 [litho_model/config.yaml](litho_model/config.yaml)。最常修改的字段如下：

| 字段 | 作用 |
|---|---|
| `PATTERN_NAME` | 输入版图名，不包含 `.bmp` 后缀 |
| `system.pixel_size_nm` | 单像素对应的物理尺寸 |
| `source.*` | 光源类型及 `sigma_in/sigma_out` |
| `resist.threshold/alpha` | 光刻胶 sigmoid 参数 |
| `meef.*` | 旧 MEEF 优化参数 |
| `sraf.*` | SRAF 分阶、参数化、宽度优化及前序控制点路径 |
| `meef_pipeline.*` | 新 MEEF 管线的 LSM 输入、控制点和迭代参数 |

例如：

```yaml
PATTERN_NAME: &PATTERN_NAME "对角通孔"
mask:
  image_name: *PATTERN_NAME
```

对应文件必须真实存在：

```text
data/patterns/对角通孔.bmp
```

不要写成 `对角通孔"`，否则双引号会成为文件名的一部分。

仓库只跟踪少量示例 BMP。批量数据、历史 LSM/OPC 结果和运行输出体积较大，不进入 Git；说明见 [data/README.md](data/README.md)。依赖历史结果的 demo 运行前还要检查：

- `sraf.meef_opt_cps_path` 指向真实的 MEEF 控制点文件；
- `meef_pipeline.ls_mask_path` 指向真实的 LSM mask；
- SRAF 初始化所需 LSM 文件与 `PATTERN_NAME` 属于同一个图形；
- mask、LSM 和控制点坐标使用一致的图像尺寸与坐标顺序。

## 3. 从哪个 demo 开始

如果只是验证环境，推荐按下面顺序运行：

```bash
# 1. 自包含、无需历史 OPC 文件的标量 SOCS
python -m scalar_socs_imaging.demo --backend numpy --grid 129 --no-plot

# 2. 数值一致性测试
python -m tests.test_batch_fft_consistency

# 3. 高 NA 矢量 SOCS 前向与反向传播
cd projects/vector_imaging
python forward_demo.py --type vector --method socs --K 20 --device cpu
cd ../..
```

`demo/scalar/case_*.py` 属于旧标量 OPC 实验链，通常需要配置文件和本地历史结果，不建议作为全新环境的第一次运行。

## 4. 根目录标量与 OPC demo

以下命令都从项目根目录执行：

| 入口 | 功能 | 主要输出或注意事项 |
|---|---|---|
| `python -m demo.scalar.case_1` | 完整的 CTM → mask 二值化/SRAF 提取 → Level-Set 优化流程 | 结果写入 `outputs/opc/case_1/`，计算时间较长 |
| `python -m demo.scalar.case_2` | 检查标量成像、PE 梯度、PV-Band、PV 梯度以及 PE+PV 联合梯度 | 会打开多幅 Matplotlib 图，主要用于算法诊断 |
| `python -m demo.scalar.case_meef` | 旧版 MEEF 矩阵 + TSVD 控制点优化，可切换 MSAA/SDF 渲染 | 依赖旧 `simulator.meef` 数据链和历史 LSM/SRAF 输入 |
| `python -m demo.scalar.case_meef_autograd` | 中心差分计算控制点梯度，再用 Adam/GD/L-BFGS 更新 | 超参数在入口文件顶部；支持 `xy` 和 `bisector` 方向 |
| `python -m demo.scalar.case_meef_pipeline` | 新 MEEF 管线：显式读取 LSM，主图形与 SRAF 都经过 CP → B-spline → MSAA | 推荐的 MEEF 实验入口；可在文件顶部选择 `tsvd` 或 `gradient` |
| `python -m demo.scalar.case_sraf` | 固定 MEEF 优化后的主图形，使用 CMA-ES 优化各阶 SRAF 半宽 | 需要 `cma`、LSM mask 和 `sraf.meef_opt_cps_path`；输出含 `run.log`、cost CSV、最优 mask/wafer |

> 当前代码状态：`LithographySimulator` 构造时会初始化 SRAF，因此所有根目录标量入口都会先读取 SRAF/LSM 历史数据；全新环境若缺少这些文件，可能在进入 `case_1` 或 `case_2` 主逻辑前就停止。旧 `case_meef` 和 `case_meef_autograd` 还要求 `simulator.meef` 已完成装配，而默认构造代码目前没有启用它。新实验优先使用 `case_meef_pipeline`；旧入口用于复现历史流程，不能视为开箱即用示例。

### `case_1`：CTM 与 Level-Set

主要流程：

```text
目标 mask → SOCS 前向 → PE 梯度 → CTM 灰度优化
          → 二值化并提取 SRAF → Level-Set → 保存 LSM
```

它用于生成后续 MEEF/SRAF 实验所需的 LSM 初始版图。

### `case_2`：PE/PV 梯度诊断

这个入口不负责完整 OPC 收敛，而是把以下量画出来并打印统计值：

- nominal aerial/wafer image；
- PE loss 与 `dPE/dMask`；
- Dose/Defocus 条件下的 PV map；
- `dPV/dMask`；
- PE 与 PV 联合损失及总梯度。

### 三个 MEEF 入口

- `case_meef`：MEEF 矩阵和 TSVD 路径，适合复现旧算法。
- `case_meef_autograd`：仍使用中心差分得到梯度，但用常见优化器更新 CP。
- `case_meef_pipeline`：输入和参数更集中，显式区分主图形 CP 与冻结的 SRAF CP，适合继续开发。

### `case_sraf`：SRAF 宽度优化

当前优化变量是每一阶 SRAF 的“半宽”，不是完整线宽。若像素尺寸为 `4 nm`，半宽 `2.5 pixel` 对应的完整 CD 约为 `20 nm`。联合目标由 PV-Band、wEPE 和 SRAF printing 惩罚组成。

## 5. 独立标量 SOCS demo

这个包不依赖旧 OPC 输出，适合验证 NumPy/PyTorch、批量前向和自动求导：

```bash
# NumPy：单张、批量、能量谱和性能对比
python -m scalar_socs_imaging.demo --backend numpy --grid 129

# PyTorch CPU + autograd
python -m scalar_socs_imaging.demo --backend torch --device cpu --grid 129

# Apple Silicon MPS
python -m scalar_socs_imaging.demo --backend torch --device mps --grid 129

# 同时测试两个后端，不弹图
python -m scalar_socs_imaging.demo --backend both --batch 8 --grid 129 --no-plot
```

可用参数：

- `--backend numpy|torch|both`：选择后端；
- `--device cpu|mps|cuda`：PyTorch 设备；
- `--batch N`：批量大小；
- `--grid N`：网格边长；
- `--svd-method randomized|full`：SOCS 分解方式；
- `--no-plot`：禁用交互式绘图。

## 6. 高 NA 矢量成像 demo

矢量子项目拥有自己的 `litho_model`，必须先进入它的目录：

```bash
cd projects/vector_imaging
```

| 入口 | 功能 | 示例命令 |
|---|---|---|
| `source_demo.py` | 生成 conventional、annular、dipole、quasar 四种照明源并检查归一化 | `python source_demo.py` |
| `forward_demo.py` | 标量/矢量 Abbe 或 SOCS 前向，随后反向传播检查 mask 梯度 | `python forward_demo.py --type vector --method socs --K 20` |
| `socs_demo.py` | 以 Vector Abbe 为参考，扫描 K 值比较 SOCS 的 NMSE、速度和中心剖面 | `python socs_demo.py` |
| `batch_simulate.py` | 从图像目录批量读取 mask，执行标量或矢量 SOCS 并保存 wafer | `python batch_simulate.py --type vector --batch 8 --limit 10` |
| `meef_batch_demo.py` | Vector SOCS + SDF + 批量中心差分 + MEEF/TSVD 控制点优化 | `python meef_batch_demo.py --iterations 20 --device cpu` |
| `visualize.py` | 可视化旧 `VectorAerialModel` 的 mask、source、pupil、X/Y/非偏振结果及剖面 | `python visualize.py` |
| `test_scalar_socs.py` | Scalar Abbe 与 Scalar SOCS 的单张、批量和 autograd 一致性检查 | `python test_scalar_socs.py` |

`forward_demo.py` 常用参数：

```bash
python forward_demo.py --type scalar --method abbe
python forward_demo.py --type scalar --method socs --K 20
python forward_demo.py --type vector --method abbe
python forward_demo.py --type vector --method socs --K 20 --device cpu
```

`batch_simulate.py` 默认读取 `projects/vector_imaging/target_images/`。这个大型数据目录不在 Git 中，运行前需要自行准备，或通过 `--input` 指向自己的 mask 目录。结果默认写入 `projects/vector_imaging/output/wafer/`。

矢量项目配置位于 [projects/vector_imaging/configs/litho_system.yaml](projects/vector_imaging/configs/litho_system.yaml)，结果写入 `projects/vector_imaging/output/`。

## 7. 名称带 `demo_` 的算法模块

下面这些文件虽然名称包含 `demo_`，但主要是被 `case_*.py` 调用的实现模块，不建议直接执行：

| 模块 | 职责 |
|---|---|
| `litho_model/demo_compute_image.py` | 标量 Abbe/SOCS 前向与 wafer sigmoid |
| `litho_model/demo_extract_key_points.py` | 轮廓关键点提取 |
| `op_model/demo_pe_ctm.py` | CTM 优化器 |
| `op_model/demo_pe_gradient.py` | PE 损失及 mask 梯度 |
| `op_model/demo_pv_band.py` | Dose/Defocus PV-Band 计算 |
| `op_model/demo_pv_gradient.py` | PV-Band 梯度及 PE+PV 联合梯度 |
| `op_model/demo_level_set.py` | Level-Set 演化与重初始化 |
| `op_model/demo_epe_wepe.py` | EPE/wEPE 采样与评价 |
| `op_model/demo_meef.py` | MEEF 矩阵、TSVD/L-curve 与 CP 更新 |
| `op_model/demo_sraf.py` | SRAF 半宽联合损失与 CMA-ES/DE/SLSQP 框架 |
| `utils_model/demo_parametric.py` | 主图形参数化曲线渲染 |
| `utils_model/demo_MSAA.py` | MSAA 抗锯齿渲染 |
| `utils_model/demo_sdf_renderer.py` | SDF + sigmoid 可微渲染 |
| `utils_model/demo_sraf_utils.py` | SRAF 分离、分阶、骨架、B-spline 和宽度渲染 |

## 8. 测试与快速检查

从项目根目录执行：

```bash
# MSAA 与 SDF 渲染一致性
python -m tests.test_sdf_vs_msaa

# 新 MEEF Pipeline 的输入装配、控制点与 SRAF block
python -m tests.test_meef_pipeline_setup

# images_simulation_v2/v3 的 FFT 数值一致性
python -m tests.test_batch_fft_consistency

# MEEF 中心差分梯度与有限差分检查
python -m tests.verify_meef_autograd_vs_fd
```

这些测试不等同于完整优化收敛验证；MEEF、SRAF 和批量矢量任务仍应使用真实输入跑完，并检查输出指标。

## 9. 输出位置

| 内容 | 默认位置 |
|---|---|
| CTM、Level-Set、MEEF、SRAF | `outputs/opc/` |
| 数值诊断与缓存 | `outputs/diagnostics/`、`outputs/cache/` |
| 矢量前向、SOCS 图和批量 wafer | `projects/vector_imaging/output/` |
| 分析脚本生成的图 | 由脚本内输入/输出路径决定 |

`outputs/`、矢量 `output/`、虚拟环境和大型数据集已加入 `.gitignore`。它们保留在本地，不会随普通 `git push` 上传。

## 10. 目录结构

```text
.
├── demo/
│   └── scalar/              # 根目录标量/OPC 入口
├── litho_model/             # 旧标量成像核心与 config.yaml
├── op_model/                # OPC 优化算法
├── utils_model/             # 几何、渲染、绘图与 project_paths.py
├── scalar_socs_imaging/     # 独立标量 SOCS 包
├── projects/
│   └── vector_imaging/      # 独立高 NA 矢量成像子项目
├── tests/                   # 数值一致性与烟雾测试
├── scripts/
│   └── analysis/            # 历史结果分析、绘图与转换脚本
├── environments/            # macOS 与 Windows 环境文件
├── data/                    # 输入 BMP 和本地大型数据
├── outputs/                 # 根目录标量/OPC 运行结果
├── docs/                    # 架构、迁移说明和论文材料
└── archive/                 # 不再由主流程引用的旧实现
```

项目级路径统一定义在 `utils_model/project_paths.py`。业务代码应使用这些常量或 `project_path()`，不要依赖当前工作目录拼接 `../` 或 Windows 反斜杠路径。

## 11. 常见问题

### `ModuleNotFoundError: No module named 'cma'`

```bash
conda activate litho
python -m pip install cma
```

正常使用 `environments/macos.yml` 创建或更新环境时会自动安装它。

### `No such file ... 对角通孔".bmp`

检查 YAML 是否写成了只有结尾引号的 `对角通孔"`。正确写法是：

```yaml
PATTERN_NAME: &PATTERN_NAME "对角通孔"
```

### 找不到 LSM 或控制点 TXT

这类文件属于前序 OPC 结果，默认不会上传 GitHub。先运行对应的 CTM/Level-Set/MEEF 流程，或者把 YAML 路径改成自己已有的真实文件。

### 程序第一次运行很慢

Matplotlib 字体缓存、SOCS 核/SVD 和 FFT 缓存都可能让首次运行明显变慢。只要没有异常且 CPU 仍持续工作，可以继续等待；完整 MEEF/SRAF 优化通常远慢于单次前向测试。
