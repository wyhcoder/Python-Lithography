# 矢量光刻成像：从物理原理到 `projects/vector_imaging` 的完整代码路径

> 本文针对本仓库的 `projects/vector_imaging/` 编写。目标不是只解释某一段公式，而是从光源发光开始，一直讲到掩模、投影物镜、三分量像面电场、部分相干叠加、SOCS 加速和光刻胶输出，并与标量成像逐项对照。
>
> 先给出最重要的模型边界：这里实现的是 **“标量薄掩模 + 高 NA 矢量投影成像”**。它考虑了物镜聚焦时的偏振旋转、横纵向电场耦合和像面 `Ex/Ey/Ez`，但没有用 RCWA/FDTD/FEM 求解三维掩模结构中的 Maxwell 散射。因此它比标量成像更完整，但还不是严格的全流程三维电磁光刻模型。

---

## 1. 一句话理解标量成像与矢量成像

- **标量成像**：把光写成一个复数场 $u(x,y)$，物镜只对它做频率截断和相位调制，强度为 $|u|^2$。
- **矢量成像**：把光写成电场向量 $\mathbf E=(E_x,E_y,E_z)$；高 NA 物镜会改变每个平面波的传播方向，所以偏振方向也必须随之旋转，并产生不能忽略的纵向分量 $E_z$。最终强度是三个正交分量的模平方之和。

二者不是“FFT 与非 FFT”的差别。两者都使用 FFT，也都可以使用 Abbe 或 SOCS。真正的差别是光学传递函数从一个标量 $H$ 变成了一个随频率变化的偏振映射矩阵 $\mathbf M_0$。

---

## 2. 整条物理成像路径

```text
扩展光源 J(σx,σy)
        │
        │ 离散为多个互不相干的 source 点 s
        ▼
每个 source 点产生一束倾斜平面波
exp[j2π(fs x + gs y)]，并带有入射 Jones 向量 (Ex_in,Ey_in)
        │
        ▼
薄掩模复透射函数 m(x,y)
入射波 × 掩模 → m(x,y) exp[j2π(fs x + gs y)]
        │
        │ FFT：从空间域进入衍射频谱
        ▼
掩模衍射谱 U_s(fx,fy)
        │
        ├── 标量路径：乘复光瞳 H(fx,fy)
        │
        └── 矢量路径：先由 M0(fx,fy) 将输入偏振投影为 Ex/Ey/Ez，
                       再乘复光瞳 H(fx,fy)
        │
        │ 对每个电场分量做 IFFT
        ▼
像面电场 Ex_s(x,y), Ey_s(x,y), Ez_s(x,y)
        │
        ▼
单源点强度 Is = |Ex_s|² + |Ey_s|² + |Ez_s|²
        │
        │ 不同 source 点之间按强度非相干求和
        ▼
空中像 I_aerial = Σs ws Is
        │
        │ Sigmoid 阈值显影模型
        ▼
wafer / resist 图像
```

这里包含两类叠加，必须区分：

1. 对同一个 source 点，掩模各处产生的复振幅先相干叠加，所以必须先求复电场，再取模平方。
2. 不同 source 点被视为互不相干，所以它们之间不能先加电场，而是分别算强度后按权重相加。

---

## 3. 符号、单位与代码张量

| 符号 | 物理含义 | 代码中的单位/形状 |
|---|---|---|
| $x,y$ | 掩模/像面空间坐标 | nm，`grid.x/y: [N,N]` |
| $f_x,f_y$ | 空间频率 | cycles/nm，`grid.Fx_2d/Fy_2d` |
| $\lambda$ | 真空波长 | nm |
| NA | 数值孔径 | 无量纲 |
| $n$ | 像方介质折射率 | 无量纲 |
| $f_{\max}=\mathrm{NA}/\lambda$ | 光瞳截止频率 | cycles/nm |
| $\sigma_x,\sigma_y$ | 归一化光源坐标 | $\sigma=f/f_{\max}$ |
| $m(x,y)$ | 薄掩模复透射率 | `[N,N]`，当前通常为实数 0/1 |
| $H(f_x,f_y)$ | 标量复光瞳 | complex `[N,N]` |
| $\mathbf j$ | 入射 Jones 向量 | $(E_x^{in},E_y^{in})$ |
| $\mathbf M_0$ | 矢量偏振传递矩阵 | complex `[3,2,N,N]` |
| $\mathbf T$ | 指定 Jones 向量后的三分量传递函数 | complex `[3,N,N]` |

`Grid` 使用中心化空间坐标和中心化频率坐标，配套的 `fft2c/ifft2c` 负责在 FFT 前后进行正确的 shift。所有频率相乘对象必须遵守同一种中心化约定。

---

## 4. 为什么高 NA 下必须从标量场升级为矢量场

### 4.1 从 Maxwell 方程得到的横向约束

均匀介质中的单色平面波可以写为

$$
\mathbf E(\mathbf r,t)=\Re\{\mathbf E_0 e^{j(\mathbf k\cdot\mathbf r-\omega t)}\}.
$$

在没有自由电荷的均匀介质中，Maxwell 方程给出

$$
\nabla\cdot\mathbf E=0
\quad\Longrightarrow\quad
\mathbf k\cdot\mathbf E_0=0.
$$

这表示电场必须垂直于自己的传播方向。低 NA 时，各条光线都近似沿 $z$ 轴传播，传播方向差别很小，可以把 $E_z$ 忽略，并假定所有频率分量具有相同横向偏振。

高 NA 时，物镜收集/聚焦的光线角度很大。对于每个光瞳坐标 $(f_x,f_y)$，传播方向都不同，因此：

- 原来的 $x/y$ 偏振必须投影到新的横截面；
- $x$ 和 $y$ 分量会互相耦合；
- 为满足 $\mathbf k\cdot\mathbf E=0$，像方会出现 $E_z$；
- 偏振方向与图形方向的相对关系会改变成像结果。

所以，标量模型在高 NA 下遗漏的不是一个简单常数修正，而是一个随 $(f_x,f_y)$ 变化的方向性变换。

### 4.2 小 NA 极限为什么会退化回标量模型

当 $\alpha,\beta\rightarrow0$ 时，$\gamma\rightarrow1$，当前代码中的矩阵在光轴上取解析极限：

$$
\mathbf M_0(0,0)=
\begin{bmatrix}
1&0\\
0&1\\
0&0
\end{bmatrix}.
$$

此时输入 $x$ 偏振仍是 $x$ 偏振，输入 $y$ 偏振仍是 $y$ 偏振，$E_z=0$。若整个光瞳都处在这种近轴范围，矢量结果便接近标量结果。

这也是“标量理论是矢量理论的近轴近似”的具体含义。

---

## 5. 第一步：空间网格与频率网格

代码入口：`litho_model/grid.py::Grid`。

对于 $N\times N$ 网格和像素尺寸 $\Delta x$：

$$
x_i=(i-\lfloor N/2\rfloor)\Delta x,
\qquad
\Delta f=\frac{1}{N\Delta x},
\qquad
f_{\mathrm{Nyquist}}=\frac{1}{2\Delta x}.
$$

网格同时决定三件事：

- 空间视场 $L=N\Delta x$；
- 频率采样间隔 $\Delta f$；
- 可表达的最高空间频率 $f_{\mathrm{Nyquist}}$。

光瞳、source、掩模频谱以及倾斜相位必须使用一致的单位。当前代码统一使用 nm 和 cycles/nm，因此倾斜相位自然写成 $2\pi f x$，不需要额外单位换算。

---

## 6. 第二步：部分相干照明源

代码入口：`litho_model/source.py::Illumination`。

光源坐标不是像面坐标，而是归一化角度/频率坐标：

$$
\sigma_x=\frac{f_s}{f_{\max}},\qquad
\sigma_y=\frac{g_s}{f_{\max}},\qquad
f_{\max}=\frac{\mathrm{NA}}{\lambda}.
$$

因此一个 source 点 $s$ 对应的物理空间频率为

$$
f_s=\sigma_{x,s}\frac{\mathrm{NA}}{\lambda},\qquad
g_s=\sigma_{y,s}\frac{\mathrm{NA}}{\lambda}.
$$

当前支持四种光源：

| 类型 | 形状 |
|---|---|
| `conventional` | 实心圆盘 |
| `annular` | $\sigma_{in}\le\rho\le\sigma_{out}$ 的圆环 |
| `dipole` | 环形径向范围内的两瓣光源 |
| `quasar` | 环形径向范围内的四瓣光源 |

代码先在每个 source cell 内进行 `upsample` 子采样，再取面积平均，以减轻圆环边界锯齿。之后乘 cell 面积并归一化：

$$
w_s\ge0,\qquad \sum_s w_s=1.
$$

注意：`upsample` 提高的是每个粗网格单元内的面积积分精度，并不会把最终独立 source 点数量扩大为 `upsample²` 倍。

---

## 7. 第三步：薄掩模与倾斜照明

当前掩模模型是一个二维透射函数 $m(x,y)$。对 source 点 $s$，入射到掩模上的倾斜平面波为

$$
r_s(x,y)=\exp\left[j2\pi(f_sx+g_sy)\right].
$$

经过薄掩模后：

$$
u_s(x,y)=m(x,y)r_s(x,y).
$$

再做二维 Fourier 变换得到衍射频谱：

$$
U_s(f_x,f_y)=\mathcal F\{u_s(x,y)\}.
$$

代码对应 `VectorForwardImaging._aerial_for_jones()` 中的：

```python
phase = 2*pi*(fs*X + gs*Y)
ramp = cos(phase) + 1j*sin(phase)
u = mask * ramp
U = fft2c(u)
```

这种写法允许 source 频率为非整数频率栅格点，比直接对光瞳做整数像素 `roll` 更准确。

### 当前掩模模型包含什么、不包含什么

包含：

- 二维振幅掩模；
- 如果调用者传入 complex tensor，也可以表达二维复透射率；
- 对 mask 的 PyTorch 自动微分。

不包含：

- 掩模厚度和侧壁；
- absorber/多层膜材料的电磁边界条件；
- TE/TM 依赖的掩模衍射；
- mask 近场的三维 Maxwell 求解。

因此，矢量化发生在投影物镜传递阶段，而不是掩模散射阶段。

---

## 8. 第四步：复光瞳——孔径、像差与离焦

代码入口：`litho_model/pupil.py::Pupil`。

### 8.1 数值孔径截止

归一化光瞳坐标为

$$
\hat f_x=\frac{f_x}{\mathrm{NA}/\lambda},\qquad
\hat f_y=\frac{f_y}{\mathrm{NA}/\lambda},\qquad
\rho^2=\hat f_x^2+\hat f_y^2.
$$

圆形孔径函数为

$$
A(f_x,f_y)=
\begin{cases}
1,&\rho\le1,\\
0,&\rho>1.
\end{cases}
$$

它的物理意义是：只有落入物镜 NA 范围的掩模衍射级次能够通过。

### 8.2 静态像差

代码用 Fringe Zernike 多项式表示波前像差：

$$
W_{static}(\rho,\theta)=\sum_i c_i Z_i(\rho,\theta),
$$

其中 $c_i$ 的单位是波长。相位因子为 $\exp(j2\pi W_{static})$。

### 8.3 物理离焦相位

离焦量为 $z$ 时，代码使用

$$
W_{defocus}=
\frac{z}{\lambda}
\left[n-\sqrt{n^2-\mathrm{NA}^2\rho^2}\right].
$$

最终复光瞳为

$$
H(f_x,f_y)=A(f_x,f_y)
\exp\left\{j2\pi[W_{static}+W_{defocus}]\right\}.
$$

标量和矢量路径共用同一个 $H$。区别在于标量路径到这里就得到全部传递函数，矢量路径还要乘偏振传递矩阵。

---

## 9. 第五步：高 NA 矢量偏振传递矩阵

代码入口：`litho_model/vector_transfer.py::VectorTransfer`。

### 9.1 像方传播方向余弦

对每个频率点，代码定义

$$
\alpha=\frac{\lambda f_x}{n},\qquad
\beta =\frac{\lambda f_y}{n},\qquad
\gamma=\sqrt{1-\alpha^2-\beta^2}.
$$

$(\alpha,\beta,\gamma)$ 是像方介质内平面波传播方向的方向余弦。由于 $\mathrm{NA}=n\sin\theta$，在光瞳边缘有

$$
\sqrt{\alpha^2+\beta^2}=\frac{\mathrm{NA}}{n}=\sin\theta_{max}.
$$

### 9.2 Mansuripur 形式的传递矩阵

当前代码使用

$$
\begin{bmatrix}
E_x\\E_y\\E_z
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
M_{xx}&M_{xy}\\
M_{yx}&M_{yy}\\
M_{zx}&M_{zy}
\end{bmatrix}}
_{\mathbf M_0(f_x,f_y)}
\begin{bmatrix}
E_x^{in}\\E_y^{in}
\end{bmatrix},
$$

其中

$$
\begin{aligned}
M_{xx}&=\frac{\beta^2+\alpha^2\gamma}{\alpha^2+\beta^2},\\
M_{xy}=M_{yx}&=\frac{\alpha\beta(\gamma-1)}{\alpha^2+\beta^2},\\
M_{yy}&=\frac{\alpha^2+\beta^2\gamma}{\alpha^2+\beta^2},\\
M_{zx}&=-\alpha,\\
M_{zy}&=-\beta.
\end{aligned}
$$

光轴处是 $0/0$ 形式，代码显式换成解析极限，而不是依赖数值除法。

这六个元素的物理含义如下：

| 元素 | 含义 |
|---|---|
| $M_{xx},M_{yy}$ | 原偏振分量沿新传播方向旋转后的主分量 |
| $M_{xy},M_{yx}$ | 高角度聚焦导致的横向交叉偏振耦合 |
| $M_{zx},M_{zy}$ | 为满足电场横向约束而产生的纵向 $E_z$ |

### 9.3 Jones 向量与最终传递函数

指定入射 Jones 向量

$$
\mathbf j=
\begin{bmatrix}E_x^{in}\\E_y^{in}\end{bmatrix}
$$

后，三分量频域传递函数是

$$
\mathbf T(f_x,f_y)=H(f_x,f_y)\,\mathbf M_0(f_x,f_y)\mathbf j.
$$

代码返回 `[3,N,N]`：

$$
\mathbf T=[T_x,T_y,T_z]^T.
$$

例如：

- `(1,0)`：线性 x 偏振；
- `(0,1)`：线性 y 偏振；
- $(1,i)/\sqrt2$：圆偏振，底层接口可表达复 Jones 分量，但 YAML 当前字段和解析类型主要按实数列表使用。

---

## 10. 第六步：单 source 点的矢量像面电场

对 source 点 $s$，掩模频谱 $U_s$ 与三分量传递函数相乘：

$$
E_{s,p}(x,y)=
\mathcal F^{-1}\left\{
U_s(f_x,f_y)T_p(f_x,f_y)
\right\},
\quad p\in\{x,y,z\}.
$$

代码张量变化为：

```text
mask                         [N,N]
ramp batch                   [Bs,N,N]
U = FFT(mask × ramp)         [Bs,N,N]
T                            [3,N,N]
U[:,None] × T[None]          [Bs,3,N,N]
E = IFFT(...)                [Bs,3,N,N]
```

三个笛卡尔分量互相正交，时间平均强度在当前相对单位下写成

$$
I_s(x,y)=|E_{s,x}|^2+|E_{s,y}|^2+|E_{s,z}|^2.
$$

代码正是对 channel 维求和：

```python
I_s = (E.real * E.real + E.imag * E.imag).sum(dim=1)
```

这里不存在 $E_xE_y^*$ 这样的交叉项，因为强度是向量与自身的 Hermitian 内积 $\mathbf E^\dagger\mathbf E$；在正交坐标基中就是三个模平方之和。

---

## 11. 第七步：Abbe 部分相干求和

所有 source 点的强度按归一化权重求和：

$$
I_{aerial}(x,y)=\sum_s w_s I_s(x,y).
$$

这就是 `VectorForwardImaging._aerial_for_jones()` 的 raw aerial 输出。

### 11.1 单偏振

若 Jones 向量固定为 $\mathbf j$：

$$
I_{\mathbf j}=\sum_s w_s
\left(
|E_{s,x}^{\mathbf j}|^2+
|E_{s,y}^{\mathbf j}|^2+
|E_{s,z}^{\mathbf j}|^2
\right).
$$

### 11.2 非偏振

当前代码用等强度、互不相干的 x/y 线偏振混合表示非偏振光：

$$
I_{unpol}=\frac12\left(I_{(1,0)}+I_{(0,1)}\right).
$$

正确顺序是：先分别求两种偏振的 raw intensity，再平均，最后进入 resist 模型。`VectorForwardImaging.forward_unpolarized()` 使用的就是这个顺序。

---

## 12. 第八步：从空中像到光刻胶图像

当前 resist 不是完整化学放大胶模型，而是可微 Sigmoid 阈值模型：

$$
I_{wafer}(x,y)=
\frac{1}{1+\exp[-a(I_{aerial}(x,y)-t_r)]}.
$$

- $t_r$：阈值；
- $a$：边沿陡峭度；
- 输出接近 0/1，方便版图优化和自动微分。

必须在术语上区分：

- `raw aerial intensity`：过 resist 前的连续光强；
- `wafer/resist image`：过 Sigmoid 后的结果。

代码中的公共 `forward()` 通常直接返回 Sigmoid 后的图，而以下私有接口返回 raw aerial：

- 矢量 Abbe：`_aerial_for_jones()`；
- 矢量 SOCS：`_aerial_socs()`；
- 标量 Abbe：`_aerial_abbe()`；
- 标量 SOCS：`_aerial_socs()`。

---

## 13. 标量成像与矢量成像的逐项差别

| 项目 | 标量成像 | 当前矢量成像 |
|---|---|---|
| 基本未知量 | 单复数场 $u$ | 三分量复电场 $(E_x,E_y,E_z)$ |
| 输入偏振 | 不区分 | Jones 向量 $(E_x^{in},E_y^{in})$ |
| 传递函数 | $H:[N,N]$ | $H\mathbf M_0\mathbf j:[3,N,N]$ |
| 横向偏振耦合 | 无 | 有 $M_{xy},M_{yx}$ |
| 纵向电场 | 假设 $E_z=0$ | 显式计算 $E_z$ |
| 单 source 强度 | $|E|^2$ | $|E_x|^2+|E_y|^2+|E_z|^2$ |
| 非偏振 | 无需区分 x/y | 计算 x/y 两个不相干基态再平均 |
| 图形方向依赖 | 只来自 mask/source | 还来自图形方向与偏振的相对关系 |
| 高 NA 适用性 | 近轴近似会逐渐失效 | 更符合高角度聚焦物理 |
| 单偏振 Abbe 场通道数 | 1 | 3 |
| 非偏振 SOCS 通道数 | 1 | 6（x-pol 三通道 + y-pol 三通道） |
| 掩模模型 | 当前为薄掩模 | 仍为同一个薄掩模，不是 3D 电磁 mask |

### 13.1 公式只差在哪里

标量：

$$
E_s=\mathcal F^{-1}\{U_sH\},
\qquad
I_s=|E_s|^2.
$$

矢量：

$$
\mathbf E_s=\mathcal F^{-1}\{U_sH\mathbf M_0\mathbf j\},
\qquad
I_s=\mathbf E_s^\dagger\mathbf E_s.
$$

可以看到，source、mask、FFT、光瞳孔径、像差、部分相干强度求和以及 resist 模型都可以共用。核心新增项就是 $\mathbf M_0\mathbf j$ 和电场 channel 维度。

### 13.2 不能用一个固定比例把标量结果修正成矢量结果

因为 $\mathbf M_0$ 随 $(f_x,f_y)$ 变化，而且不同 mask 图形含有不同的空间频谱。偏振、source 形状、图形方向、NA 和像差共同决定差异。不存在对所有图形都正确的“标量强度乘某个常数”方案。

---

## 14. SOCS 为什么可以加速矢量部分相干成像

### 14.1 从 Abbe 到 TCC

Abbe 方法直接遍历 $N_s$ 个 source 点。Hopkins 形式则把部分相干系统写成频率对之间的 Transmission Cross Coefficient（TCC）。TCC 是一个 Hermitian 半正定算子，可以做特征分解：

$$
\mathrm{TCC}=\sum_k \lambda_k\,\boldsymbol\phi_k\boldsymbol\phi_k^\dagger.
$$

于是强度可以写成多个相干系统强度之和：

$$
I(x,y)\approx
\sum_{k=1}^{K}
\left\|
\mathcal F^{-1}\{M(f)\boldsymbol\Phi_k(f)\}
\right\|_2^2,
$$

其中 $K\ll N_s$ 时便能加速重复成像。

### 14.2 当前代码不显式构造巨大 TCC

对每个 source 点，把它的系统传递函数拉直并乘 $\sqrt{w_s}$，组成矩阵 $A$：

$$
A=[\sqrt{w_1}\,\mathbf t_1,
   \sqrt{w_2}\,\mathbf t_2,\ldots,
   \sqrt{w_{N_s}}\,\mathbf t_{N_s}],
\qquad
\mathrm{TCC}=AA^\dagger.
$$

然后直接做 thin SVD：

$$
A=U\Sigma V^\dagger.
$$

则

$$
AA^\dagger=U\Sigma^2U^\dagger,
\qquad
\lambda_k=\sigma_k^2.
$$

代码把奇异值直接吸收到核里：

$$
\boldsymbol\Phi_k=\sigma_k\mathbf u_k.
$$

因此前向时只需要求 $|\cdot|^2$ 后相加，不再额外乘 $\lambda_k$。

### 14.3 标量、单偏振矢量、非偏振矢量的矩阵尺寸

| 模式 | $A$ 的行数 | SOCS 核形状 |
|---|---:|---|
| 标量 | $N^2$ | `[K,1,N,N]` |
| 矢量单偏振 | $3N^2$ | `[K,3,N,N]` |
| 矢量非偏振 | $6N^2$ | `[K,6,N,N]` |

非偏振版本的前 3 个 channel 对应 x 偏振产生的 $(E_x,E_y,E_z)$，后 3 个对应 y 偏振。前向求完 6 个 channel 的强度和后应乘 $1/2$。

### 14.4 SOCS 与矢量化是两个正交概念

- 矢量化回答“每个相干系统里传播几个电场分量、偏振如何变换”；
- SOCS 回答“如何把许多 source 点压缩成少数相干核”。

所以四种组合都存在：标量 Abbe、矢量 Abbe、标量 SOCS、矢量 SOCS。

---

## 15. `projects/vector_imaging` 的真实调用路径

### 15.1 Demo 入口

入口文件是 `projects/vector_imaging/forward_demo.py`：

```text
main()
  ├─ SimulationConfig.from_yaml(...)
  ├─ build_optical_system(...)
  │    ├─ Grid(...)
  │    ├─ Pupil(...)
  │    ├─ Illumination(...)
  │    └─ VectorTransfer(...)
  ├─ 根据 --type 选择 scalar / vector
  ├─ 根据 --method 选择 abbe / socs
  ├─ make_demo_mask(...)
  ├─ forward + loss + backward
  └─ plot_results(...)
```

运行选择关系：

```text
--type vector --method abbe
    → VectorForwardImaging

--type vector --method socs
    → VectorSOCS → VectorSOCSImaging

--type scalar --method abbe
    → ScalarForwardImaging

--type scalar --method socs
    → ScalarSOCS → ScalarSOCSImaging
```

注意：当前 CLI 的 `--type` 默认值是 `scalar`，尽管文件说明写的是“矢量成像 demo”。若要明确运行矢量路径，应显式写 `--type vector`。

### 15.2 各源文件职责

| 文件 | 职责 |
|---|---|
| `configs/litho_system.yaml` | 波长、像素、NA、折射率、source、Jones 向量、resist 参数 |
| `litho_model/config.py` | YAML → dataclass |
| `litho_model/grid.py` | 空间/频率网格 |
| `litho_model/mask.py` | 图像读取、归一化、居中 |
| `litho_model/pupil.py` | NA 孔径、Zernike 像差、离焦、复光瞳 |
| `litho_model/source.py` | 部分相干 source 分布与归一化权重 |
| `litho_model/vector_transfer.py` | $[3,2]$ 的高 NA 偏振矩阵及 Jones 投影 |
| `litho_model/forwardImage.py` | 标量/矢量 Abbe 与 SOCS 前向、Sigmoid |
| `litho_model/socs.py` | 构造 source 传递矩阵并做 SVD |
| `litho_model/lithography_simulator.py` | 将配置组装为 grid/mask/pupil/source/vector transfer |
| `tool/fft_tool.py` | 中心化 FFT/IFFT 与频率轴工具 |

### 15.3 矢量 Abbe 的代码数据流

```text
Illumination.get_axes/get_weights
    │  σ坐标与 ws
    ▼
VectorForwardImaging._build_active_source_points
    │  fs=σx·NA/λ, gs=σy·NA/λ
    ▼
VectorTransfer.transfer_for_jones
    │  T = H · M0 · j                 [3,N,N]
    ▼
VectorForwardImaging._aerial_for_jones
    │  ramp                           [Bs,N,N]
    │  U=FFT(mask·ramp)               [Bs,N,N]
    │  E=IFFT(U[:,None]·T[None])      [Bs,3,N,N]
    │  Is=Σchannel |E|²               [Bs,N,N]
    │  aerial=Σsource ws Is           [N,N]
    ▼
sigmoid(aerial)
    │
    ▼
wafer image                            [N,N]
```

### 15.4 矢量 SOCS 的代码数据流

预计算阶段：

```text
T0 = H · M0 · j                       [3,N,N]
t0 = IFFT(T0)                         [3,N,N]
对每个 source 点 s：
    Ts = FFT(t0 · ramp_s)             [3,N,N]
    A[:,s] = sqrt(ws) · flatten(Ts)   [3N²]
A = U S Vᴴ
kernels = reshape(Sk Uk)              [K,3,N,N]
```

重复前向阶段：

```text
M = FFT(mask)                         [N,N]
E = IFFT(M · kernels)                 [K,C,N,N]
aerial = ΣK,C |E|²                    [N,N]
```

SOCS 的优势主要发生在同一光学系统要反复计算大量 mask 的场景，例如 OPC、ILT 或 MEEF 优化。核预计算较重，但之后每张 mask 不再逐 source 点计算。

---

## 16. 当前 YAML 对应的物理系统

`configs/litho_system.yaml` 当前配置为：

| 参数 | 当前值 | 物理解释 |
|---|---:|---|
| 波长 | 193 nm | ArF 波段 |
| NA | 1.35 | 高 NA 浸没式条件 |
| 像方折射率 | 1.44 | 使 NA 可以大于 1 |
| $N$ | 257 | 257×257 仿真网格 |
| 像素尺寸 | 4 nm | 视场 $257\times4=1028$ nm |
| Jones | `[1,0]` | x 线偏振 |
| source | annular | 环形照明 |
| $\sigma_{in},\sigma_{out}$ | 0.6, 0.9 | source 圆环范围 |
| 静态像差 | `{4: 0.05}` | 当前代码会把 Fringe Z4 以 0.05 波加入静态波前 |
| 物理离焦 | 0 nm | YAML 的 `defcous_nm` 拼写由解析器兼容 |

由这些数值可得：

$$
f_{max}=\frac{1.35}{193}\approx0.006995\ \mathrm{cycles/nm},
$$

$$
\Delta f=\frac{1}{257\times4}\approx0.000973\ \mathrm{cycles/nm},
$$

$$
\frac{\Delta f}{f_{max}}\approx0.139.
$$

因此 source 的最终粗坐标步长约为 $0.139\sigma$。`upsample: 10` 能改善每个 cell 的圆环覆盖率，但不改变这个最终 source 点间距。

最大像方会聚半角满足

$$
\sin\theta_{max}=\frac{NA}{n}=\frac{1.35}{1.44}=0.9375,
$$

即 $\theta_{max}\approx69.6^\circ$。这已经明显超出近轴条件，使用随角度变化的偏振投影是合理的；但 $E_z$ 的实际强度比例必须针对具体 mask/source 数值计算，不能仅由 NA 给出一个普适百分比。

---

## 17. 与仓库根目录标量实现的关系

仓库里有两套可作为“标量基线”的代码：

1. `projects/vector_imaging/litho_model/forwardImage.py` 内的 `ScalarForwardImaging`、`ScalarSOCS`、`ScalarSOCSImaging`。它与矢量版本共用 `Grid/Pupil/Illumination`，最适合做严格的 scalar-vs-vector 对比。
2. 根目录的 `litho_model/lithography_simulator.py` 和 `scalar_socs_imaging/`。它们也是标量 SOCS 思路，但数据结构、归一化和 source shift 实现不完全相同。

特别是根目录旧标量 `litho_model/lithography_simulator.py` 使用 `np.roll` 将连续 source 坐标舍入为整数频率像素位移；`projects/vector_imaging` 则通过空域 phase ramp 表达 source shift，可以保留亚像素频移。若两个目录产生差异，不能把所有差异都归因于“标量 vs 矢量”，还要先统一采样、归一化、shift 方法、K、阈值和 resist 参数。

最干净的比较方式是只在 `projects/vector_imaging` 内比较：

```text
相同 Grid
相同 Pupil
相同 Illumination
相同 mask
相同 raw aerial 接口
只把 H 替换成 H·M0·j
```

---

## 18. 静态代码审查中发现的实现边界与注意事项

下面这些不会改变前面的物理原理，但会影响当前程序输出的解释和 scalar/vector 数值比较。

### 18.1 `forward()` 返回的不是 raw aerial

`VectorForwardImaging.forward()`、`ScalarForwardImaging.forward()` 和两个 SOCS imaging 的 `forward()` 都会经过 Sigmoid。`forward_demo.py` 中变量名 `aerial_x/aerial_y/aerial_unpol` 容易让人误以为它们是 raw aerial，实际上多数路径已是 resist 输出。

做物理强度比较时应调用 raw 接口，或为类增加正式的 `forward_aerial()` 公共接口。

### 18.2 Abbe demo 的非偏振顺序不严格

demo 当前执行：

```python
aerial_x = forward.forward(mask, [1, 0])  # 已过 sigmoid
aerial_y = forward.forward(mask, [0, 1])  # 已过 sigmoid
aerial_unpol = 0.5 * (aerial_x + aerial_y)
```

这相当于“先显影再平均”。物理上应当“先平均 raw intensity 再显影”，即使用 `forward.forward_unpolarized(mask)`。

`forward_batch_unpolarized()` 也通过 `_aerial_batch()` 得到已过 Sigmoid 的结果后再平均，存在同样问题。

### 18.3 非偏振 Vector SOCS 的 batch 路径缺少 $1/2$

单 mask 的 `VectorSOCSImaging._aerial_socs()` 在 kernel channel 数为 6 时会乘 `0.5`；但 `VectorSOCSImaging.forward_batch()` 直接对 6 channel 求和后进入 Sigmoid，没有对应的 `0.5`。因此非偏振 6-channel kernel 的单张和 batch 结果可能不一致。

### 18.4 YAML resist 阈值没有自动贯穿 demo

YAML 写的是 `threshold: 0.25, alpha: 85`，但 Abbe 类内部 hardcode 为 `threshold=0.15, alpha=85`，SOCS demo 构造时也使用默认 `0.15`，没有把 `cfg.resist.sigmoid_params` 传进去。

### 18.5 `energy_fraction()` 当前不能报告真实截断能量

`VectorSOCS`/`ScalarSOCS` 只保存前 $K$ 个奇异值，之后 `energy_fraction()` 又用这同一组截断值同时计算分子和分母，所以通常会得到 1。真实能量保留率需要保存全部奇异值，或用 $\|A\|_F^2$ 作为总能量分母。

### 18.6 注释说“随机 SVD”，实际使用完整 SVD

`projects/vector_imaging/litho_model/socs.py` 当前调用 `torch.linalg.svd(A, full_matrices=False)`，这是 thin/full-spectrum SVD，不是随机化 SVD。矩阵较大时，预计算速度和内存应按实际实现评估。

### 18.7 Z4 注释与当前行为不一致

`Pupil._normalize_aberrations()` 的注释说 Z4 由专用 defocus 接口处理并跳过，但构造函数当前没有调用这个规范化函数，而是直接使用传入字典。因此 YAML 的 `{4: 0.05}` 目前确实会作为静态 Zernike Z4 加入，同时还可以叠加物理 `defocus_nm`。

### 18.8 当前矢量光瞳仍是简化模型

代码中的 $H$ 是所有偏振分量共享的标量复光瞳，$\mathbf M_0$ 负责几何偏振投影。当前没有额外建模：

- lens 的 Jones pupil / 偏振像差；
- s/p 不同的 Fresnel 透射；
- 双折射和去偏振；
- 显式 apodization 因子；
- 空间变化的 source 偏振分布；
- 三维 resist 内的 standing wave、吸收、扩散和 PEB。

因此，更准确的名称是“高 NA 矢量薄掩模空中像模型”。

### 18.9 YAML 的 Jones 配置没有贯穿 demo

`system.jones_matrix` 会被 `SimulationConfig` 读入，但 `forward_demo.py` 的实际矢量前向仍显式写死 `(1,0)` 和 `(0,1)`，没有使用 `cfg.system.jones_matrix`。因此只修改 YAML 的 Jones 值，目前不会自动改变 demo 的单偏振结果。

### 18.10 CLI 默认值与帮助文字不一致

`--method` 的实际默认值是 `socs`，但帮助字符串仍说 Abbe 是默认；`--type` 的实际默认值是 `scalar`。复现实验时应在命令中显式给出 `--type vector --method abbe|socs`，不要依赖文字说明推断默认路径。

### 18.11 没有显式的投影缩小倍率

网格中的 `pixel_size_nm` 被直接作为成像平面的采样尺寸，代码里没有单独的 4× reduction/magnification 参数。因此输入 BMP 应理解为已经在当前仿真坐标尺度上的透射图，而不是直接把 reticle 尺寸原样放入后再由程序自动缩小。

---

## 19. 建议的 scalar-vs-vector 对比方法

为了只测量矢量物理本身的影响，建议遵守以下实验控制：

1. 使用同一份 `Grid/Pupil/Illumination/mask`。
2. 比较 raw aerial，不比较已经过 Sigmoid 的图。
3. 矢量使用 `0.5*(Ix_raw+Iy_raw)`；标量使用同一 source 权重。
4. Abbe 先作为参考，避免 SOCS 截断误差混入。
5. 比较前不要分别做 max normalization，否则会抹掉总强度差异。
6. 同时报告：NMSE、最大绝对误差、总能量、中心截线、阈值轮廓/CD。
7. 再扫描 NA、偏振、图形旋转角和 source 类型，观察差异如何变化。

推荐分解误差来源：

```text
标量 Abbe vs 矢量 Abbe
    → 纯 scalar/vector 模型差异

矢量 Abbe vs 矢量 SOCS(K)
    → SOCS 截断误差

标量 Abbe vs 标量 SOCS(K)
    → 标量 SOCS 截断误差

raw aerial vs sigmoid wafer
    → resist 非线性放大的差异
```

---

## 20. 最终总结

这套矢量成像的主线可以压缩成下面一个公式：

$$
I_{aerial}(x,y)=
\sum_s w_s
\left\|
\mathcal F^{-1}
\left\{
\mathcal F[m(x,y)e^{j2\pi(f_sx+g_sy)}]
\;H(f_x,f_y)\;
\mathbf M_0(f_x,f_y)\;\mathbf j
\right\}
\right\|_2^2.
$$

从左到右依次是：

1. source 点给掩模施加倾斜照明；
2. 掩模生成衍射频谱；
3. 复光瞳筛选频率并加入像差/离焦；
4. 高 NA 矩阵把输入 Jones 偏振变成像面 $E_x/E_y/E_z$；
5. IFFT 得到像面矢量电场；
6. 三分量模平方求和得到单 source 强度；
7. 不同 source 点按强度非相干求和；
8. 最后通过 Sigmoid 得到简化 wafer 图像。

标量模型就是把上式中的 $\mathbf M_0\mathbf j$ 和三分量范数去掉，只保留一个复场。高 NA 下，由于传播方向变化大，这个被去掉的部分正是偏振旋转、交叉偏振和纵向场的来源。

---

## 21. 参考文献

1. M. Mansuripur, “Distribution of light at and near the focus of high-numerical-aperture objectives,” *JOSA A* 3, 2086–2093 (1986). [Optica 页面](https://opg.optica.org/abstract.cfm?uri=josaa-3-12-2086)，DOI: `10.1364/JOSAA.3.002086`。
2. B. Richards and E. Wolf, “Electromagnetic diffraction in optical systems. II. Structure of the image field in an aplanatic system,” *Proceedings of the Royal Society A* 253, 358–379 (1959). DOI: [10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)。
3. H. H. Hopkins, “On the diffraction theory of optical images,” *Proceedings of the Royal Society A* 217, 408–432 (1953). DOI: [10.1098/rspa.1953.0071](https://doi.org/10.1098/rspa.1953.0071)。
4. A. E. Rosenbluth et al., “Fast calculation of images for high numerical aperture lithography,” *Proc. SPIE* (2004). [IBM Research 摘要](https://research.ibm.com/publications/fast-calculation-of-images-for-high-numerical-aperture-lithography)。
