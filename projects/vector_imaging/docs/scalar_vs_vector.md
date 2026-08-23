# 标量成像 vs 矢量成像

> 本文档总结 `projects/vector_imaging` 中两套光刻成像模型的差异：
> **标量** (`ScalarForwardImaging` / `ScalarSOCS` / `ScalarSOCSImaging`)
> **矢量** (`VectorForwardImaging` / `VectorSOCS` / `VectorSOCSImaging`)。

---

## 1. 一句话总结

|       | 标量 (Scalar)               | 矢量 (Vector)                                                  |
| ----- | ------------------------- | ------------------------------------------------------------ |
| 物理近似  | 把光当成单分量复振幅 $u(x,y)$       | 把光当成三分量电场 $\mathbf{E}=(E_x, E_y, E_z)$（含偏振+高 NA 投影）            |
| 适用场景  | NA ≲ 0.7 的低 NA 系统、教学/原型验证 | NA ≳ 0.9 的高 NA 浸没式光刻（NA=1.35, n=1.44），偏振依赖、Ez 分量不可忽略         |
| 计算成本  | **低**（~3× 更快，1 个分量）       | 高（3 或 6 个分量同时算）                                              |
| 数值精度  | 在低 NA 下与矢量基本一致；在 NA=1.35 下系统性偏离 | 物理严谨                                                         |

---

## 2. 物理建模差异

### 2.1 光学传递函数

#### 标量

只有一个**复光瞳函数**：

$$
H(f_x, f_y) = \text{aperture}(f_x, f_y) \cdot \exp\!\bigl[\,j\,2\pi \cdot W(f_x, f_y)\bigr]
$$

- `aperture` 为 NA 限频圆
- $W$ 为 Zernike 像差 + 离焦相位
- **形状**：`[N, N]`，单分量复张量

代码：`pupil.pupil`（在 `pupil.py` 中构建）

#### 矢量

在标量光瞳基础上，叠加 **Mansuripur 偏振传递矩阵** $\mathbf{M}_0$：

$$
\begin{bmatrix} E_x \\ E_y \\ E_z \end{bmatrix} =
\underbrace{\begin{bmatrix}
M_{xx} & M_{xy} \\
M_{yx} & M_{yy} \\
M_{zx} & M_{zy}
\end{bmatrix}}_{\mathbf{M}_0(f_x, f_y)\ \in\ \mathbb{C}^{3\times 2\times N \times N}}
\cdot \begin{bmatrix} E_x^{\text{in}} \\ E_y^{\text{in}} \end{bmatrix}
$$

各元素由介质内方向余弦 $\alpha = \lambda f_x / n$、$\beta = \lambda f_y / n$、$\gamma = \sqrt{1 - \alpha^2 - \beta^2}$ 决定：

$$
M_{xx} = \frac{\beta^2 + \alpha^2 \gamma}{\rho^2}, \quad
M_{xy} = M_{yx} = \frac{\alpha\beta(\gamma - 1)}{\rho^2}, \quad
M_{yy} = \frac{\alpha^2 + \beta^2\gamma}{\rho^2}, \quad
M_{zx} = -\alpha, \quad M_{zy} = -\beta
$$

最终的矢量传递函数：

$$
T_p(f) = \bigl[\mathbf{M}_0(f) \cdot \mathbf{j}\bigr]_p \cdot H(f), \quad p \in \{x, y, z\}
$$

- $\mathbf{j} = (E_x^{\text{in}}, E_y^{\text{in}})$ 为入射 Jones 向量
- **形状**：`[3, N, N]` 复张量（每个偏振分量一个通道）

代码：`vector_transfer.py::VectorTransfer.transfer_for_jones()`

### 2.2 单源点的强度

设 source 点 $s$ 的物理空间频率为 $(f_s, g_s)$，权重 $w_s$，倾斜照明：

$$
\text{tilted}(x, y) = \text{mask}(x, y) \cdot \exp\bigl[j\,2\pi(f_s x + g_s y)\bigr]
$$

#### 标量

$$
E_s(x) = \mathcal{F}^{-1}\bigl\{\mathcal{F}\{\text{tilted}\} \cdot H(f)\bigr\}, \quad
I_s(x) = |E_s(x)|^2
$$

#### 矢量

$$
E_{s,p}(x) = \mathcal{F}^{-1}\bigl\{\mathcal{F}\{\text{tilted}\} \cdot T_p(f)\bigr\}, \quad p \in \{x, y, z\}
$$

$$
I_s(x) = |E_{s,x}|^2 + |E_{s,y}|^2 + |E_{s,z}|^2
$$

**关键差异**：矢量版有 **3 倍 IFFT** 开销（每个偏振分量一次），而且 $|E_z|^2$ 在高 NA 下贡献显著。

### 2.3 Abbe 求和（部分相干合成）

两者形式一致：

$$
\text{aerial}(x) = \sum_{s=1}^{N_{\text{src}}} w_s \cdot I_s(x)
$$

只是 $I_s$ 的算法不同。

非偏振情况下，矢量版要算两次：

$$
I_{\text{unpol}} = \tfrac{1}{2}\bigl[I_x(\mathbf{j}=(1,0)) + I_y(\mathbf{j}=(0,1))\bigr]
$$

### 2.4 光刻胶模型

两者完全一致：

$$
\text{wafer} = \sigma\bigl(\alpha (\text{aerial} - tr)\bigr)
= \frac{1}{1 + e^{-\alpha(\text{aerial} - tr)}}
$$

---

## 3. SOCS 加速差异

SOCS（Sum of Coherent Systems）的核心思想：把 $N_{\text{src}}$ 个源点的传递函数堆成矩阵 $A$，做截断 SVD 取前 $K$ 项，把每张 mask 的成像从 $N_{\text{src}}$ 次 FFT 对压缩为 $K$ 次（$K \ll N_{\text{src}}$）。

### 3.1 A 矩阵尺寸

| 模式            | A 形状                | 物理含义                     |
| ------------- | ------------------- | ------------------------ |
| 标量            | $[N^2,\ N_{\text{src}}]$ | 每列：1 个源点的频域传递函数（拉直后）     |
| 矢量（单偏振）       | $[3N^2,\ N_{\text{src}}]$ | 每列：3 个偏振分量的传递函数堆叠        |
| 矢量（非偏振）       | $[6N^2,\ N_{\text{src}}]$ | 进一步拼接 x、y 两次单偏振的 A       |

代码位置：
- `socs.py::ScalarSOCS._build_socs_kernel`（标量）
- `socs.py::VectorSOCS._build_socs / _build_socs_unpolarized`（矢量）

### 3.2 SOCS 核形状

| 模式      | 核形状           | 内存（N=257, K=20） |
| ------- | ------------- | -------------- |
| 标量      | `[K, 1, N, N]`  | ~10 MB         |
| 矢量（单偏振） | `[K, 3, N, N]`  | ~30 MB         |
| 矢量（非偏振） | `[K, 6, N, N]`  | ~60 MB         |

### 3.3 前向公式

```
# 标量 (forwardImage.py::ScalarSOCSImaging)
M(f) = FFT2(mask)                          # [N, N]
MK   = M[None,None] * kernels              # [K, 1, N, N]
E    = IFFT2(MK)                           # [K, 1, N, N]
I    = (|E|²).sum(dim=(0, 1))              # [N, N]

# 矢量 (forwardImage.py::VectorSOCSImaging)
M(f) = FFT2(mask)                          # [N, N]
MK   = M[None,None] * kernels              # [K, C, N, N], C=3 or 6
E    = IFFT2(MK)                           # [K, C, N, N]
I    = (|E|²).sum(dim=(0, 1))              # [N, N]
if C == 6: I *= 0.5                        # 非偏振归一化
```

广播张量大小（关键性能指标）：

| 模式            | $|E|^2$ 大小（B=1）           |
| ------------- | ------------------------ |
| 标量            | $[K \times N^2]$ 实数       |
| 矢量（单偏振）       | $[K \times 3 \times N^2]$ |
| 矢量（非偏振）       | $[K \times 6 \times N^2]$ |

矢量版多 3 ~ 6 倍内存与算力。

---

## 4. 代码接口对照表

| 功能                                | 标量                                           | 矢量                                                |
| --------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| Abbe 前向类                          | `ScalarForwardImaging`                       | `VectorForwardImaging`                            |
| SOCS 加速类                          | `ScalarSOCS` + `ScalarSOCSImaging`           | `VectorSOCS` + `VectorSOCSImaging`                |
| 单 mask 接口                         | `forward(mask) -> [N, N]`                    | `forward(mask, jones=(1, 0)) -> [N, N]`           |
| 非偏振接口                             | （没必要，结果一致）                                   | `forward_unpolarized(mask)`                       |
| Batch 接口                          | `forward_batch(masks) -> [B, N, N]`          | `forward_batch(masks)` / `forward_batch_unpolarized(masks)` |
| 是否过 sigmoid                       | 是                                            | 是                                                 |
| Raw aerial 接口                     | `forward_batch_aerial(masks)` (不过 sigmoid)   | 内部 `_aerial_socs(mask)` 或 `_aerial_for_jones(...)` |
| 偏振参数                              | 无                                            | `jones: Tuple[Ex, Ey]` 或 `unpolarized=True`       |
| 必需依赖                              | `Grid`, `Pupil`, `Illumination`              | 上述 + `VectorTransfer`                             |

---

## 5. 数值对比（实测）

测试条件：N=257, dx=8 nm, λ=193 nm, NA=1.35, 环形光源 σ∈[0.6, 0.9]，K=20。

### 5.1 标量 SOCS vs 标量 Abbe

测试脚本：`projects/vector_imaging/test_scalar_socs.py`

| 指标         | 数值       |
| ---------- | -------- |
| NMSE (单 mask) | 8.6e-5   |
| max\_abs\_err | 4.4e-3   |
| 能量保留率      | 100%     |
| autograd 链路 | ✓ 全有限非零 |

### 5.2 矢量 vs 标量（同一 mask）

理论上的差异主要来源：
1. **Ez 分量贡献**：高 NA 下不可忽略，直接影响 $I = |E_x|^2 + |E_y|^2 + |E_z|^2$ 大小
2. **偏振调制**：$M_{xx}, M_{yy}$ 在大角度下偏离 1
3. **Ez 占比**与 NA 关系：

| NA   | $|E_z|^2 / |E|^2$ 估计 |
| ---- | ------------------- |
| 0.5  | ~5%                 |
| 0.9  | ~15%                |
| 1.35 | **~25%**             |

**结论**：在 ArF 浸没式光刻 (NA=1.35) 下，**必须用矢量模型**；标量结果会系统性偏低 ~20%。

---

## 6. 何时选哪个？

```
┌─ NA < 0.7 (DUV 干式低 NA, 教学/快速原型) ─→ 标量足够
│
├─ 0.7 ≤ NA ≤ 0.9 (干式高 NA) ─────────────→ 视精度需求；偏振敏感场景用矢量
│
├─ NA > 0.9 (浸没式) ──────────────────────→ 必须矢量
│
└─ 自动微分 / 梯度优化 (OPC, MEEF) ─────────→ 两者都支持，但矢量更准
```

### 6.1 在 MEEF 优化中的取舍

`projects/vector_imaging/litho_model/meef_opt/` 目前用的是 `VectorSOCSImaging`。如果只是在低 NA 下做算法验证，可以替换成 `ScalarSOCSImaging`：

```python
# 在 meef_batch_demo.py 的 build_optical_system 中
from litho_model.socs import ScalarSOCS
from litho_model.forwardImage import ScalarSOCSImaging

# 替换 VectorSOCS / VectorSOCSImaging
socs = ScalarSOCS(pupil=pupil, illumination=illum, K=20, weight_threshold=1e-6)
imaging = ScalarSOCSImaging(socs=socs, grid=grid,
                             sigmoid_tr=cfg.resist.sigmoid_params.threshold,
                             sigmoid_a=cfg.resist.sigmoid_params.alpha)
```

`meef_opt/optimizer.py` 中的 `compute_aerial_batch` 已经兼容 `[K, C, N, N]` 中 C=1/3/6 三种情况，无需改动。

---

## 7. 参考文献

- Mansuripur, M. *"Distribution of light at and near the focus of high-numerical-aperture objectives"*. JOSA A 3(12), 1986.
- Hopkins, H.H. *"On the diffraction theory of optical images"*. Proc. R. Soc. Lond. A 217(1130), 1953.
- Adam, K. *"SOCS Decomposition for Photolithography Modeling"*. Proc. SPIE 5754, 2005.

---

## 附录：核心代码片段对比

### 单源点强度计算的差异

```python
# 标量 (forwardImage.py::ScalarForwardImaging._aerial_abbe)
H = self.pupil.pupil.to(cdtype)              # [N, N] 复光瞳
U = fft2c(mask_c.unsqueeze(0) * ramp)        # [B, N, N]
E = ifft2c(U * H.unsqueeze(0))               # [B, N, N] 单分量
I = (E.real**2 + E.imag**2)                  # [B, N, N]

# 矢量 (forwardImage.py::VectorForwardImaging._aerial_for_jones)
T = self.vector_transfer.transfer_for_jones(...)  # [3, N, N] 三分量
U = fft2c(mask_c.unsqueeze(0) * ramp)             # [B, N, N]
UT = U.unsqueeze(1) * T.unsqueeze(0)               # [B, 3, N, N]
E = ifft2c(UT)                                     # [B, 3, N, N]
I = (E.real**2 + E.imag**2).sum(dim=1)             # [B, N, N] 三分量加和
```

差异本质：**矢量版多了一个 channel 维度，每个 channel 是一个偏振分量**。
