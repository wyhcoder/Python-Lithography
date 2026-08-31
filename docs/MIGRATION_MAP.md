# 目录迁移映射

本次整理使用可逆的目录移动，没有删除实验结果。

| 原位置 | 新位置 | 说明 |
|---|---|---|
| `case_*.py` | `demo/scalar/` | 标量与 OPC 示例入口 |
| `Vector_Image_code/` | `projects/vector_imaging/` | 独立矢量子项目 |
| `pattern_data/` | `data/patterns/` | 输入版图 |
| `TXT_test/` | `data/debug_dumps/scalar_text/TXT_test/` | 历史数值转储 |
| 根目录的 `TXT_test\\*.txt` 假路径文件 | `data/debug_dumps/legacy_windows_paths/` | 规范化为真实目录和文件名 |
| `illum.txt` | `data/debug_dumps/vector/illum.txt` | 历史矢量光源转储 |
| `OPCproject/` | `outputs/opc/` | OPC 历史结果与后续输出 |
| `test/test_*.py` | `tests/` | 可判定的测试/验证脚本 |
| `test/` 中的绘图和诊断脚本 | `scripts/analysis/` | 手工分析工具 |
| `test/` 中的 PNG/PDF/SVG/CSV | `outputs/diagnostics/` | 历史诊断产物 |
| `overleaf_wepe/` 与 zip/tex | `docs/papers/` | 论文与 Overleaf 材料 |
| `output.log` | `outputs/logs/output.log` | 历史运行日志 |
| `cache/imresize_cache.pkl` | `outputs/cache/imresize_cache.pkl` | 历史缓存 |
| 三份根目录环境 YAML | `environments/` | 按 macOS、Windows full/min 明确命名 |
| 根目录 `demo_select_eps.py` | `utils_model/select_eps.py` | 当前主流程实际使用的实现 |
| 旧 `utils_model/demo_select_eps.py` | `archive/legacy_code/select_eps_legacy.py` | 未被主流程引用的旧版本 |
| 硬编码 Windows 盘符的 `test/sigmoid.py` | `archive/legacy_code/sigmoid_cpp_python_compare.py` | 仅保留作历史 C++/Python 对照 |

兼容性修正：

- `utils_model/project_paths.py` 统一解析项目根、输入和输出目录；
- 标量示例可从任意工作目录直接运行；
- YAML 中的旧结果路径已改为 `outputs/opc/...`；
- 矢量配置中的相对 mask 目录固定相对于矢量子项目解析；
- 调试矩阵不再写到根目录。
