# Data directory

Git 仓库只保存 `patterns/` 顶层的少量示例 BMP，确保默认标量案例可以运行。

以下大型或可再生成内容仅保存在本地，不进入普通 Git 历史：

- `debug_dumps/`：历史中间矩阵与调试转储；
- `patterns/target_images/`：批量版图数据集；
- `patterns/target_images_副本/`：历史备份；
- `../projects/vector_imaging/target_images/`：矢量批处理使用的数据副本。

发布数据集时应打包为版本化归档，并记录下载地址和 SHA256；不要直接取消
`.gitignore` 后把数千个数据文件提交进源码仓库。
