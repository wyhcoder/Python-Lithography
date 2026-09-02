import numpy as np

from .simulation_parameters import SimulationParameters
from .load_mask_bmp import load_mask_image
from .set_patterns import Mask
from .set_optical_system import OpticalSystem
from .set_illumination import Illumination
from .set_meef import MEEF
from .set_sraf import SRAF
from utils_model.plot_matrix import plot_matrix, plot_psf
from matplotlib import pyplot as plt

class LithographySimulator:
    """光刻仿真器主类.

    职责:
      - 根据配置组装四大组件: Mask / OpticalSystem / Illumination / SRAF;
      - 维护优化过程中的版图状态 (op_mask) 与不变量缓存 (opt_cache);
      - 通过 prepare_for_optimization() 预计算 PSF/SOCS 核, 供前向仿真复用.
    """
    def __init__(self, params: SimulationParameters):
        """根据配置对象一次性装配所有仿真组件."""
        print("--- Initializing Lithography Simulator ---")
        self.params = params

        # 1. Mask: 从 BMP 读入二值版图, 像素尺寸由 system.pixel_size_nm 决定
        init_mask_data = load_mask_image(self.params.mask.image_name)
        self.mask = Mask(initial_mask_data=init_mask_data, 
                         pixel_size=self.params.system.pixel_size_nm)
        print("Mask component created.")

        # 2. OpticalSystem: 投影物镜, 含 NA / 波长 / 像差; 决定光瞳函数
        self.optics = OpticalSystem(na=self.params.optics.na, 
                                    wavelength_nm=self.params.system.wavelength_nm, 
                                    grid_size=self.mask.grid_size,
                                    pitch=self.mask.pitch,
                                    aberrations=self.params.optics.aberrations)
        # print("OpticalSystem component created.")

        # 3. Illumination: 光源, 由 sigma_in/sigma_out 决定环形/部分相干分布
        self.source = Illumination(source_type=self.params.source.type, 
                                   sigma_in=self.params.source.sigma_in, 
                                   sigma_out=self.params.source.sigma_out,
                                   frequency_coords=self.optics.frequency_coords)
        print("Illumination component created.")
        
        
        # 4. SRAF: 亚分辨率辅助图形, 含骨架/控制点/分阶等配置, 供 SRAF_Optimizer 使用
        # self.sraf = SRAF(target_mask=self.mask.data,
        #                  curve_type=self.params.sraf.curve_type,
        #                  pattern_name=self.params.sraf.pattern_name,
        #                  file_name=self.params.sraf.file_name,
        #                  sraf_orders=self.params.sraf.sraf_orders,
        #                  fix_sraf=self.params.sraf.fix_sraf,
        #                  epe_cost=self.params.sraf.epe_cost,
        #                  pvband_cost=self.params.sraf.pvband_cost,
        #                  sraf_simplify=self.params.sraf.sraf_simply,
        #                  meef_opt_cps_path=self.params.sraf.meef_opt_cps_path)
        # print("SRAF component created.")
        
        # 5. MEEF 优化组件 (case_meef.py / case_meef_autograd.py 依赖此).
        # self.meef = MEEF(target_mask=self.mask.data,
        #                  curve_type=self.params.meef.curve_type,
        #                  simplt_type=self.params.meef.simply_type,
        #                  val_simply=self.params.meef.val_simply,
        #                  pattern_name=self.params.meef.pattern_name,
        #                  file_name=self.params.meef.file_name,
        #                  ifOPC=self.params.meef.ifOPC,
        #                  SRAF=self.params.meef.SRAF,
        #                  move_strategy=self.params.meef.move_strategy)
        # print("MEEF component created.")
        
        
        
        # 6. 优化状态:
        #   op_mask    当前正在优化的 mask (None 时使用初始 mask);
        #   opt_cache  prepare_for_optimization() 写入的不变量缓存
        #              (key, H_k_list, source_weights, H_k_fft_stack, ...)
        self.op_mask = None
        self.opt_cache = {}
        
        print("--- Simulator Initialized Successfully ---")
        
        
    def prepare_for_optimization(self, method : str = "SOCS"):
        """预计算优化期间固定不变的量, 写入 self.opt_cache.

        必须在优化循环开始前调用一次. 重复调用同一光照+光学系统时自动跳过.

        Args:
            method: 前向仿真方案
                "Abbe" - Abbe 方法: 每个光源点单独计算 PSF, 再加权求和;
                "SOCS" - Sum-of-Coherent-Systems: 对 A 矩阵做 SVD, 保留前
                         K 个相干核, 大幅加速 (推荐).

        缓存内容:
            key:             (id(optics), id(source)) 用于幂等校验;
            H_k_list:        PSF / SOCS 核列表 (空间域);
            source_weights:  Abbe -> 光源强度; SOCS -> 特征值 λ_i;
            norm_factor:     SOCS 模式的归一化因子 (Σ source_weights);
            H_k_fft_stack:   H_k 的 batch FFT, 给 images_simulation_v3 加速用.
        """
        print("\n--- Preparing for optimization by pre-computing constants (H_k, etc.) ---")
        
        # 幂等保护: 同一组光学+光源不重复计算
        cache_key = (id(self.optics), id(self.source))
        if self.opt_cache.get("key") == cache_key:
            print("Constants are already cached.")
            return

        # ---- 共用准备: 把光源点的 (k_x, k_y) 转成像素级频移 ----
        pupil_func = self.optics.pupil_function
        source_map = self.source.source_map
        
        # 只保留有强度的光源点 (阈值 1e-9 去掉浮点噪声)
        sindy, sindx = np.nonzero(source_map > 1e-9)
        source_weights = source_map[sindy, sindx]   # 一维: 每个有效光源点的强度
        
        # 空间频率步长 (= 光瞳 / specimen 网格步长)
        pupil_k_step = self.optics.frequency_coords[1] - self.optics.frequency_coords[0]
        source_k_coords_x = self.source.source_coords[sindx]
        source_k_coords_y = self.source.source_coords[sindy]
        # 把连续频率坐标量化为整数像素位移, 用于 np.roll 频移光瞳
        shifts_x = np.round(source_k_coords_x / pupil_k_step).astype(int)
        shifts_y = np.round(source_k_coords_y / pupil_k_step).astype(int)
        if method == "Abbe":
            # ---- Abbe: 每个光源点 -> 一个频移光瞳 -> 一个 PSF ----
            H_k_list = []
            for sy, sx in zip(shifts_y, shifts_x):
                # 在光瞳上频移 (sy, sx) 像素
                shifted_pupil = np.roll(pupil_func, (sy, sx), axis=(0, 1))
                # 频域 -> 空间域 (含 fftshift 把零频归中)
                H_k = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(shifted_pupil)))
                H_k_list.append(H_k)

            self.opt_cache = {
                "key": cache_key,
                "H_k_list": H_k_list,                # 每个光源点的 PSF (空间域)
                "source_weights": source_weights,    # 对应的光源强度
            }
            # 额外预算: 一次性 batch FFT, 给 v3 前向仿真加速
            self._build_H_k_fft_stack(H_k_list)
            print(f"Pre-computation finished. Cached {len(H_k_list)} PSFs.")
        
        if method == "SOCS":
            # ---- SOCS: 对 A 矩阵 SVD, 用前 K 个相干核近似 Abbe 总和 ----
            # 保留的相干核数 (经验值, 与像素尺寸相关)
            if self.mask.pixel_size == 4.0:
                num_kernels = 50  # pixel_size=4nm 取 50
            else:
                num_kernels = 60  # pixel_size=6nm 取 60
            N = pupil_func.shape[0]
            M = len(source_weights)
            
            # 1) 组装 A: 每列 = sqrt(w_s) · P(k - k_s), 拉平成 N² 维列向量
            A = np.zeros((N * N, M),dtype= complex)
            print(f"Constructing A matrix: {N*N} rows x {M} columns...")
            for i in range(M):
                shifted_p = np.roll(pupil_func,(shifts_y[i], shifts_x[i]), axis=(0,1))
                A[:,i] = np.ravel(np.sqrt(source_weights[i]) * shifted_p)
            
            # 2) SVD: A = U Σ Vᴴ
            #    U:  (N²,  M)  左奇异向量 (每列对应一个相干核的频域表示)
            #    S:  (M,)      奇异值 (降序)
            #    Vh: (M, M)    右奇异向量
            U, S, Vh = np.linalg.svd(A, full_matrices=False)

            # 3) 特征值 λ_i = σ_i², 取前 K 个核
            lambdas = S**2
            
            socs_kernels_spatial = []
            actual_k = min(num_kernels, M)
            
            for i in range(actual_k):
                # U 的第 i 列还原成 N×N 的频域核
                kernel_freq = U[:, i].reshape((N, N))
                # 频域 -> 空间域 (光刻前向仿真用空间域卷积)
                kernel_spatial = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel_freq)))
                socs_kernels_spatial.append(kernel_spatial)
            
            # 注: 早期试过 sum(λ·|Σ_kernel|²) 归一化, 与原版差异过大, 已弃用.
            self.opt_cache={
                "key" :cache_key,                          # 幂等校验键
                "H_k_list": socs_kernels_spatial,           # 前 K 个 SOCS 相干核 (空间域)
                "source_weights": lambdas[:actual_k],       # 对应的特征值 λ_i
                "norm_factor": np.sum(source_weights),      # 归一化因子
            }
            # 额外预算: batch FFT 加速版前向仿真
            self._build_H_k_fft_stack(socs_kernels_spatial)
            print("normal_factor",np.sum(source_weights))
            print(f"SOCS done. Top {actual_k} kernels capture "
            f"{np.sum(lambdas[:actual_k])/np.sum(lambdas)*100:.2f}% energy."
            f"The length of H_k_list is {len(socs_kernels_spatial)}")
            # self.plot_socs_results(lambdas=lambdas, kernels=socs_kernels_spatial,num_to_show=6)
      
    
        
    def _build_H_k_fft_stack(self, H_k_list):
        """
        把 H_k_list 堆成 (N, H, W) 的 ndarray 并预先做 fft2，写入 opt_cache。

        - 这一步只在 prepare_for_optimization 里执行一次，是迭代过程中的常量。
        - images_simulation_v3 会优先读取 H_k_fft_stack 走 batch FFT 路径，
          相比 N 次单 FFT 卷积可显著加速；缓存缺失时 v3 自动 fallback 到老路径。
        - 与原本 fftconvolve1 的实现保持一致：使用 np.fft.fft2 (corner-origin)，
          mask 端也对应做 fft2，因此不需要额外的 ifftshift。
        - 数学等价：单次 fft2(H_k) ≡ batch fft2(stack, axes=(-2,-1))[i]，仅有
          浮点累加顺序的微小差异，对下游 round(., 6) 完全无影响。
        """
        if not H_k_list:
            return
        try:
            # 堆叠为 (N, H, W) 复数数组
            H_k_stack = np.stack(H_k_list, axis=0).astype(np.complex128, copy=False)

            # 一次性做 batch FFT；axes=(-2,-1) 在最后两维上做 2D FFT
            H_k_fft_stack = np.fft.fft2(H_k_stack, axes=(-2, -1))

            # 写入缓存（不动 H_k_list，保持向后兼容）
            self.opt_cache["H_k_fft_stack"] = H_k_fft_stack

            # 友好日志：让用户知道额外占用的内存
            mem_mb = H_k_fft_stack.nbytes / (1024 * 1024)
            print(f"  H_k_fft_stack cached: shape={H_k_fft_stack.shape}, "
                  f"dtype={H_k_fft_stack.dtype}, mem={mem_mb:.1f} MB")
        except MemoryError:
            # 内存不够时不阻塞主流程，v3 会自动 fallback 到 H_k_list 循环
            print("  [WARN] Not enough memory for H_k_fft_stack; "
                  "images_simulation_v3 will fall back to per-kernel loop.")

    def update_mask(self, new_op_mask: np.ndarray):
        """更新当前优化中的 mask. shape 必须与初始 mask 一致."""
        print("\n--- Updating simulator with new op_mask ---")
        if new_op_mask.shape != self.mask.data.shape:
            raise ValueError(f"The shape of new_op_mask {new_op_mask.shape} does not match the simulator's grid shape {self.mask.data.shape}")
        self.op_mask = new_op_mask
    
    def get_current_mask_spatial(self) -> np.ndarray:
        """返回当前应使用的空间域 mask: 优先 op_mask, 否则回退到初始 mask."""
        if self.op_mask is not None:
            return self.op_mask
        else:
            return self.mask.data

    def view_current_mask(self, title: str = "Current Mask"):
        """可视化当前 mask, 标题自动标注是否为优化后版本."""
        current_mask = self.get_current_mask_spatial()
        
        if self.op_mask is not None:
            plot_matrix(current_mask, f"{title} (Optimized)")
        else:
            plot_matrix(current_mask, f"{title} (Initial)")
            

    def plot_socs_results(self,lambdas, kernels, num_to_show=6):
        """可视化 SOCS 结果: 上图特征值能量分布 (Scree Plot), 下图前 K 个相干核振幅."""
        # 计算能量占比
        total_energy = np.sum(lambdas)
        cumulative_energy = np.cumsum(lambdas) / total_energy * 100
        
        # 创建画布 (增加一行用于显示特征值分布曲线)
        num_to_show = min(num_to_show, len(kernels))
        cols = 3
        rows = (num_to_show + cols - 1) // cols + 1 # 多加一行给频谱图
        
        fig = plt.figure(figsize=(15, 5 * rows))
        
        # --- 1. 绘制特征值下降曲线 (Scree Plot) ---
        ax_energy = fig.add_subplot(rows, 1, 1)
        ax_energy.bar(range(1, len(lambdas) + 1), lambdas / total_energy, color='skyblue', label='Individual')
        ax_energy.step(range(1, len(lambdas) + 1), cumulative_energy / 100, where='mid', color='red', label='Cumulative')
        ax_energy.set_title("Eigenvalue Spectrum (Energy Distribution)")
        ax_energy.set_ylabel("Normalized Energy")
        ax_energy.set_xlabel("Kernel Index")
        ax_energy.set_yscale('log') # 逻辑坐标能更清楚看到微小特征值
        ax_energy.grid(True, which="both", ls="-", alpha=0.5)
        ax_energy.legend()

        # --- 2. 绘制各个核 ---
        for i in range(num_to_show):
            ax = fig.add_subplot(rows, cols, i + cols + 1) # 从第二行开始画
            
            # 计算振幅 (Amplitude)
            mag = np.abs(kernels[i])
            
            # 归一化显示，方便看清形状
            im = ax.imshow(mag, cmap='magma', interpolation='bilinear')
            
            # 核心部分：在标题中显示特征值及其能量占比
            energy_pct = (lambdas[i] / total_energy) * 100
            ax.set_title(f"Kernel {i+1}\n$\lambda$ = {lambdas[i]:.2e}\n({energy_pct:.2f}%)")
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')

        plt.tight_layout()
        plt.show()