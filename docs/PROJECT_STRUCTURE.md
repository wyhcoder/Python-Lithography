# 项目框架说明

## 1. 为什么这个项目看起来像“几套代码叠在一起”

它确实包含三条相对独立的成像链路，以及建立在旧标量链路上的 OPC 优化层：

| 部分 | 数值框架 | 主要职责 | 入口 |
|---|---|---|---|
| 旧标量成像 | NumPy / SciPy | Mask、光源、光瞳、Abbe/SOCS、光刻胶 | `demo/scalar/case_*.py` |
| OPC 优化层 | NumPy / SciPy / OpenCV | CTM、Level-Set、MEEF、PV-Band、SRAF | `op_model/` |
| 独立标量 SOCS | NumPy / PyTorch | 自包含 SOCS 与批量/autograd 对照 | `scalar_socs_imaging/demo.py` |
| 高 NA 矢量成像 | PyTorch | Jones 入射偏振、三分量电场、Vector Abbe/SOCS | `projects/vector_imaging/*.py` |

根目录旧标量包与矢量子项目中都存在名为 `litho_model` 的包，这是历史形成的命名
冲突。矢量代码因此保持为独立子项目，并应从它自己的目录启动。

## 2. 旧标量主线的数据流

```text
litho_model/config.yaml
        │
        ▼
SimulationParameters.from_yaml()
        │
        ▼
LithographySimulator
  ├── load_mask_image() <── data/patterns/<image_name>.bmp
  ├── Mask
  ├── OpticalSystem
  └── Illumination
        │
        ▼
prepare_for_optimization("Abbe" | "SOCS")
        │
        ▼
images_simulation_v2/v3
        │
        ├── CTM / Level-Set
        ├── MEEF / MEEF Pipeline
        ├── PV-Band
        └── SRAF
                │
                ▼
            outputs/opc/
```

其中 `litho_model` 负责“给定 mask 算出 aerial/wafer image”，`op_model` 负责“根据
损失和梯度更新 mask 或控制点”，`utils_model` 提供几何参数化、渲染和落盘工具。

## 3. 两套 SOCS 为什么不能直接合并

`litho_model/` 中的 SOCS 是旧标量优化流程的一部分，接口围绕
`LithographySimulator.opt_cache` 设计。`scalar_socs_imaging/` 则是自包含库，接口围绕
`OpticalConfig` 和 NumPy/PyTorch 后端设计。二者可做数值对照，但目前不是同一套 API。

## 4. 矢量子项目的数据流

```text
projects/vector_imaging/configs/litho_system.yaml
        │
        ▼
Grid + Pupil + Illumination + VectorTransfer
        │
        ├── VectorForwardImaging (Abbe)
        └── VectorSOCSImaging
                │
                ▼
projects/vector_imaging/output/
```

这里的 `litho_model` 只属于矢量子项目。它使用 PyTorch，并显式处理入射 Jones 向量与
像面 `Ex/Ey/Ez` 分量；它不是根目录旧标量包的替代文件夹。

## 5. 目录职责边界

- `data/` 是输入或保留的历史数值数据，不写运行结果。
- `outputs/` 是可重新生成的输出、缓存和日志。
- `demo/` 只负责组装与启动，不放核心算法。
- `tests/` 放可判定通过/失败的检查。
- `scripts/analysis/` 放需要人工指定历史结果路径的绘图、诊断和转换工具。
- `docs/` 放说明与论文材料。
- `archive/` 中的代码不再被主流程导入，仅用于追溯。

## 6. 下一阶段建议

当前整理优先保证路径与既有导入兼容。若继续做源码级重构，建议顺序为：

1. 把现有脚本改成 pytest，并为三条成像链路分别建立小尺寸基准数据；
2. 将 `demo_` 前缀从核心算法文件中移除；
3. 给旧标量代码建立唯一顶层包名，解决两个 `litho_model` 的冲突；
4. 最后再迁移到 `src/` 布局并加入 `pyproject.toml`。
