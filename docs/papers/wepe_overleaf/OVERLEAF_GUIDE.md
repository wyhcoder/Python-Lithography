# Overleaf 编译指南

## 现象
报错：
```
! Critical Package ctex Error: CTeX fontset `fandol' is unavailable in current mode.
!pdfTeX error: pdflatex (file unisong61): Font unisong61 at 657 not found
```

## 原因
- Overleaf 默认编译器是 **pdfLaTeX**。
- `ctex` 包的默认字体集 `fandol` 只支持 **XeLaTeX / LuaLaTeX**，在 pdfLaTeX 下不可用。
- 文件首行 `% !TEX program = xelatex` 在 Overleaf 里**不会被自动识别**，必须手动设置。

## 解决方案（任选其一）

### 方案 A：切换编译器（推荐，中文效果最好）
1. 打开 Overleaf 项目
2. 左上角 **Menu** 按钮
3. **Compiler** → 选 **XeLaTeX**
4. 点 **Recompile**

### 方案 B：不切换编译器（兜底方案）
- 本次已用 `\ifPDFTeX` 自动判断编译器：pdfLaTeX 走 `CJKutf8` 兜底，**也能成功编译**。
- 中文排版效果略简化（无中文标点挤压、粗体略弱），但内容完全正确。
- 适用场景：只关心内容、不在意排版细节。

## 重新上传步骤
如果你想完全从头来过：
1. 把 `docs/papers/wepe_overleaf/` 文件夹压缩为 zip
2. Overleaf → New Project → **Upload Project** → 选 zip
3. 进入后按方案 A 或 B 操作

## 文件结构
```
wepe_overleaf/
├── wepe_direction_compare.tex   (主文件，双引擎兼容)
├── OVERLEAF_GUIDE.md            (本指南)
└── figures/                     (10 张 PNG)
    ├── dps_bisector_mask.png / dps_bisector_wafer.png
    ├── dps_xy_mask.png        / dps_xy_wafer.png
    ├── pipe_bisector_mask.png / pipe_bisector_wafer.png
    ├── pipe_xy_mask.png       / pipe_xy_wafer.png
    ├── loss_dps.png
    └── loss_pipe.png
```
