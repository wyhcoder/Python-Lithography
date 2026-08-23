# Litho Simulation CPU

这是一个计算光刻仿真与 OPC 优化工作区。仓库内包含三套成像实现和一组共享的
OPC 实验代码；它们的物理目标相近，但实现框架并不相同，不能把代码目录直接混用。

## 先看整体框架

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
- `scalar_socs_imaging/`：自包含的标量 SOCS 库，提供 NumPy 与 PyTorch 两个后端。
- `projects/vector_imaging/`：独立的高 NA 偏振矢量成像子项目，拥有自己的配置、模型、demo、数据和输出。

更详细的模块职责和数据流见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 整理后的目录

```text
.
├── litho_model/             # 旧标量成像核心
├── op_model/                # OPC 优化算法
├── utils_model/             # 共享工具与 project_paths.py
├── scalar_socs_imaging/     # 独立标量 SOCS 包
├── projects/
│   └── vector_imaging/      # 独立矢量成像子项目
├── examples/
│   └── scalar/              # 标量/OPC 可运行案例
├── tests/                   # 数值一致性与烟雾测试
├── scripts/
│   └── analysis/            # 绘图、诊断、转换脚本
├── environments/            # macOS 与 Windows 环境文件
├── data/
│   ├── patterns/            # 输入 BMP 版图
│   └── debug_dumps/         # 历史数值转储
├── outputs/
│   ├── opc/                 # 旧 OPCproject 的全部结果
│   ├── diagnostics/         # 测试图、调试矩阵
│   ├── cache/               # 运行缓存
│   └── logs/                # 历史日志
├── docs/                    # 项目说明与论文材料
└── archive/                 # 不再由主流程引用的旧实现
```

四个核心 Python 包暂时仍保留在根目录，这是为了保持现有绝对导入兼容；后续如果要
做第二阶段包名重构，应先建立完整自动化测试，再迁移到 `src/` 布局。

## 环境

Apple Silicon macOS 推荐：

```bash
conda env create -f environments/macos.yml
conda activate litho
```

若本机已经存在 `litho` 环境，可直接激活，不要重复创建。

`environments/windows-full.yml` 和 `environments/windows-min.yml` 含 Windows 平台依赖，
不要在 macOS 上使用。

## 常用运行方式

所有标量命令都可以在项目根目录执行：

```bash
python -m examples.scalar.case_1
python -m examples.scalar.case_2
python -m examples.scalar.case_meef
python -m examples.scalar.case_meef_autograd
python -m examples.scalar.case_meef_pipeline
python -m examples.scalar.case_sraf
```

独立标量 SOCS：

```bash
python -m scalar_socs_imaging.demo --backend numpy --grid 129
```

矢量模型：

```bash
cd projects/vector_imaging
python forward_demo.py --type vector --method socs --K 20
python batch_simulate.py --type vector --batch 8 --limit 10
```

快速检查：

```bash
python -m tests.test_sdf_vs_msaa
python -m tests.test_meef_pipeline_setup
python -m tests.test_batch_fft_consistency
```

## 路径约定

项目级路径统一定义在 `utils_model/project_paths.py`。业务代码应使用这些常量或
`project_path()`，不要再拼接 `Path.cwd()`、`../` 或平台相关的反斜杠路径。

- 输入只放 `data/`；
- 程序生成物只放 `outputs/`；
- 可复用源码放四个核心包；
- 可运行案例放 `examples/`；
- 一次性分析工具放 `scripts/analysis/`；
- 独立实现放 `projects/`。

原目录到新目录的完整映射见 [docs/MIGRATION_MAP.md](docs/MIGRATION_MAP.md)。
