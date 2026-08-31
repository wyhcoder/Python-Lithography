import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v2
from op_model.demo_epe_wepe import caculate_epe, caculate_wepe
from utils_model.demo_MSAA import AntiAliasRenderer
from utils_model.demo_parametric import ParametricDemo
from utils_model.demo_meef_utils import save_excel, get_cp_vectors, highlight_contour_red, _save_images, drawcostcurve, show_sample_mcps
from utils_model.demo_sraf_utils import save_width_results, read_cps_txt, distance_msaa_multi_skeleton, distance_msaa_multi_curves, extract_sraf
from op_model.demo_pv_band import PVBandComputer

class _TeeStream:
    """同时写到多个底层 stream 的轻量封装 (用于把控制台输出复制到文件).

    与直接 reassign sys.stdout 相比, 用 _TeeStream 的好处:
      - 屏幕仍正常显示, 不影响交互体验;
      - 同时把每一行 print 的内容追加到日志文件;
      - 文件内容用 flush() 立即落盘, 避免长任务异常退出后丢日志.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                # 单个 stream 失败不应影响其它 (例如文件系统满)
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    # 让 isatty 等查询沿用第一个 stream (通常是真实 terminal),
    # 避免某些库把 tee 误识别成非交互式后改变行为.
    def isatty(self) -> bool:
        first = self._streams[0] if self._streams else None
        return bool(first and getattr(first, "isatty", lambda: False)())

    def fileno(self) -> int:
        # 部分库 (如 matplotlib) 可能调用 fileno; 用第一个真实 stream 的
        first = self._streams[0]
        return first.fileno()


@contextmanager
def _tee_console_to_file(log_path: Path):
    """上下文管理器: 进入时把 sys.stdout/sys.stderr 复制一份写到 log_path.

    用法:
        with _tee_console_to_file("dir/run.log"):
            print("xxx")    # 同时打印到屏幕和文件
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 'w' 覆盖写, 一次 run 一份日志 (如需追加改 'a' 即可).
    f = open(log_path, "w", encoding="utf-8", buffering=1)  # 行缓冲
    f.write(f"# SRAF Optimizer Run Log\n")
    f.write(f"# Started at: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# Working dir: {Path.cwd()}\n")
    f.write("# " + "-" * 60 + "\n")
    f.flush()

    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(real_stdout, f)
    sys.stderr = _TeeStream(real_stderr, f)
    try:
        yield log_path
    finally:
        # 复位 + 收尾
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        f.write("# " + "-" * 60 + "\n")
        f.write(f"# Finished at: "
                f"{datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
        f.close()


class  SRAF_Optimizer:
    def __init__(self, simulator: LithographySimulator) :
        """
        初始化优化器。

        Args:
            simulator: 配置好的 LithographySimulator 实例。

        Note:
            self.iterations 由 simulator.sraf.pvband_cost 决定:
              - pvband_cost=True  → 1 次  (联合 cost 模式, 全局优化器一次
                                 性给出 4D 最优宽度向量, 不需要外层迭代);
              - pvband_cost=False → 50 次 (旧 EPE 模式, 沿用控制点+宽度
                                 单步梯度下降的迭代式优化, 当前已废弃).
        """
        self.simulator = simulator
        if simulator.sraf.pvband_cost:
            self.iterations = 1     # 联合 cost 一次性优化, 不需要外层循环
        else:
            self.iterations = 50    # 旧 EPE 模式 (废弃)
        self.target_mask = simulator.mask.data.copy()   # 优化目标是原始的二值掩模
        self.render = AntiAliasRenderer(msaa_level=16)  # 使用 MSAA 渲染器
        self.parametric = ParametricDemo(self.simulator.sraf.curve_type,
                                         self.target_mask) # 拟合成为参数化曲线
        self.pv_computer = PVBandComputer(
                            simulator=simulator,
                            dose_margin=0.05,      # ±5% 剂量容差
                            dof_range_nm=200.0,    # ±100nm 离焦范围
                            dof_steps=3,           # 采样 -100, 0, +100
                            mode='full',           # Dose + Defocus 全模式 dose_only
                            method="SOCS"
                            )

    def run(self):
        """对外入口. 自动把控制台输出 tee 到输出目录的 run.log 文件.

        实际优化逻辑全部在 _run_impl() 中, run() 仅负责日志包装.
        如果不想要日志文件, 直接调用 self._run_impl() 即可.
        """
        out_dir = Path(self.simulator.sraf.filepath)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        with _tee_console_to_file(log_path):
            print(f"[log] 控制台输出同步保存至: {log_path}")
            self._run_impl()

    def _run_impl(self):
        #1.初始化参数
        initial_mask = self.simulator.sraf.target_mask
        mask_shape = initial_mask.shape
        #2.准备预计计算缓存和追踪变量
        # self.simulator.prepare_for_optimization()
        current_mask = self.simulator.sraf.LSM_mask       #CTM level set 优化后的mask

        # 计算水平集优化后的误差
        ai_lsm, wafer_lsm, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )
        print(f"[sraf_opt]存储位置：{self.simulator.sraf.filepath}")
        print(f"[sraf_opt]初始sraf宽度：",self.simulator.sraf.initial_width)
        _save_images(ai_lsm, wafer_lsm, current_mask,self.target_mask,self.simulator.sraf.filepath,"ls版图", pixel_size=self.simulator.mask.pixel_size)
        ls_epe, ls_epe_vector = caculate_epe(ai_lsm, self.simulator.params.resist.threshold,self.simulator.sraf.eps,self.simulator.sraf.num_eps,
                                       self.simulator.sraf.weight_meef,add_weight=False)
        ls_pe = np.sum((self.target_mask - wafer_lsm) ** 2)
        ls_wepe = caculate_wepe(self.simulator.sraf.wepe_caculated, ls_epe_vector, self.simulator.sraf.num_weps)
        print('#######################################################################')
        print(f"[LSM-ILT]LS后的平均wEPE误差：{ls_wepe/self.simulator.sraf.num_weps:.6f}，平均EPE误差：{ls_epe/self.simulator.sraf.num_eps:.6f}，PE误差：{ls_pe:.6f}")

        # highlight_contour_red(self.target_mask, wafer_lsm, save_path=f"{self.simulator.sraf.filepath}/ls_wi.png")
        LSM_save_dir = Path(self.simulator.sraf.filepath)   # 目录
        save_path =  LSM_save_dir / "LSM_wepe_result.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"LSM_EPE误差为 {ls_wepe/self.simulator.sraf.num_weps:.6f}, "
                f"LSM_wEPE误差为 {ls_epe/self.simulator.sraf.num_eps:.6f}\n"
            )


        # 记录器
        epe_errors = []
        pe_errors = []
        iterations = []
        wepe_errors = []
        iteration_time = []
        pvband_errors = []
        #   第一个if分支需要舍弃，当前没用到
        if self.simulator.sraf.pvband_cost == False:
            current_cps = self.simulator.sraf.cps
            current_mask = self.parametric.render_curve(current_cps) + self.simulator.sraf.srafsin_iniwidth
            ai_initial, wafer_initial, intermediates = images_simulation_v2(
                    mask_spatial=current_mask,
                    resist_params=self.simulator.params.resist,
                    opt_cache=self.simulator.opt_cache
                )
            initial_epe, initial_epe_vector = caculate_epe(ai_initial, self.simulator.params.resist.threshold,self.simulator.sraf.eps,self.simulator.sraf.num_eps,
                                        self.simulator.sraf.weight_meef,add_weight=False)
            initial_wepe = caculate_wepe(self.simulator.sraf.wepe_caculated, initial_epe_vector, self.simulator.sraf.num_weps)
            initial_pe = np.sum((self.target_mask - wafer_initial) ** 2)
            print(f"拟合后初始平均wEPE误差：{initial_wepe/self.simulator.sraf.num_weps:.6f}，拟合后初始EPE误差：{initial_epe/self.simulator.sraf.num_eps:.6f},拟合后初始PE误差：{initial_pe:.6f}")

            epe_errors.append(ls_epe/self.simulator.sraf.num_eps)
            pe_errors.append(ls_pe)
            wepe_errors.append(ls_wepe/self.simulator.sraf.num_weps)
            iterations.append(0)
            iteration_time.append(0)
            pvband_errors.append(0)
            #绘制初始控制点以及ep点分布
            show_sample_mcps(self.target_mask, self.target_mask, self.simulator.sraf.cps,
                            self.simulator.sraf.eps,self.simulator.sraf.filepath, self.simulator.sraf.weight_meef)

            #初始化控制点的位置初始宽度SRAFs以及优化器初始值
            current_cps = self.simulator.sraf.cps.copy()
            current_vectors = [get_cp_vectors(cps) for cps in current_cps]
            # current_srafs = self.simulator.sraf.srafsin_iniwidth.copy()
            current_sraf_width =  self.simulator.sraf.original_width

        else:
            # 这个部分是读取case_meef 优化后的主图形的控制点坐标
            # 因为我们的优化路径是先优化主图形，再优化SRAF，所以这里需要读取主图形优化后的控制点坐标
            # cps_txt_path 从 simulator.sraf 配置读取，避免硬编码路径
            raw_path = self.simulator.sraf.meef_opt_cps_path
            if not raw_path:
                raise FileNotFoundError(
                    "config.yaml 中 sraf.meef_opt_cps_path 未配置 (空字符串)。\n"
                    "请填写 MEEF 优化后主图形控制点 TXT 文件的相对/绝对路径。"
                )
            cps_txt_path = Path(raw_path)
            if not cps_txt_path.is_file():
                raise FileNotFoundError(
                    f"主图形优化后的控制点文件不存在或不是文件: {cps_txt_path}\n"
                    f"请在 config.yaml 的 sraf.meef_opt_cps_path 中配置正确路径。"
                )
            current_cps = read_cps_txt(str(cps_txt_path))


            # 主图形在优化sraf宽度时保持不变
            # 计算主图形优化后的误差
            # read_cps_txt 返回 ragged list[ndarray(N_i,2)]，不要外层强转 np.array
            # print(current_cps)
            current_mask = self.parametric.render_curve(current_cps) + self.simulator.sraf.srafsin_iniwidth
            ai_epe_opt, wafer_epe_opt, intermediates = images_simulation_v2(
                    mask_spatial=current_mask,
                    resist_params=self.simulator.params.resist,
                    opt_cache=self.simulator.opt_cache
                )
            epe_opt, epe_opt_vector = caculate_epe(ai_epe_opt, self.simulator.params.resist.threshold,self.simulator.sraf.eps,self.simulator.sraf.num_eps,
                                        self.simulator.sraf.weight_meef,add_weight=False)
            pe_opt = np.sum((self.target_mask - wafer_epe_opt) ** 2)
            wepe_opt = caculate_wepe(self.simulator.sraf.wepe_caculated, epe_opt_vector, self.simulator.sraf.num_weps)
            pvband_opt_main,_ = self.pv_computer.compute(current_mask)
            print(  f"[MEEF_opt]main区域用MEEF矩阵优化后的wEPE:{wepe_opt/self.simulator.sraf.num_weps:.6f}\n"
                    f"[MEEF_opt]EPE:{epe_opt/self.simulator.sraf.num_eps:.6f}\n"
                    f"[MEEF_opt]PE:{pe_opt:.6f}\n"
                    f"[MEEF_opt]pvband:{pvband_opt_main:.6f}")


            # 记录优化结果
            epe_errors.append( epe_opt/self.simulator.sraf.num_eps)
            pe_errors.append(pe_opt)
            wepe_errors.append(wepe_opt/self.simulator.sraf.num_weps)
            iterations.append(0)
            iteration_time.append(0)
            pvband_errors.append(pvband_opt_main)
            current_vectors = [get_cp_vectors(cps) for cps in current_cps]
            # current_srafs = self.simulator.sraf.srafsin_iniwidth.copy()


            # if self.simulator.sraf.fix_sraf:
            #     dim_x = self.simulator.sraf.num_cps
            # else:
            #     dim_x = self.simulator.sraf.num_cps + self.simulator.sraf.num_srafs
            current_sraf_width =  self.simulator.sraf.original_width

            epe_opt_save_dir = Path(self.simulator.sraf.filepath)   # 目录
            save_path =  epe_opt_save_dir / "主图形EPE优化后result.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(
                f"mainMEEF矩阵优化后的wEPE误差为 {wepe_opt/self.simulator.sraf.num_weps:.6f}\n"
                f"EPE误差为 {epe_opt/self.simulator.sraf.num_eps:.6f}\n"
                f"pvband为 {pvband_opt_main:.6f}\n")

        # ---------- WEPE 最优 ----------
        best_wepe = np.inf
        best_wepe_epe = None
        best_wepe_mask = None
        best_wepe_wafer = None
        best_wepe_it = None
        best_wepe_pvband = None
        # ---------- EPE 最优 ----------
        best_epe = np.inf
        best_epe_wepe = None
        best_epe_mask = None
        best_epe_wafer = None
        best_epe_it = None

        # 在循环外初始化 current_SRAFs，防止 pvband_cost=False 路径下 NameError
        current_SRAFs = self.simulator.sraf.srafsin_iniwidth.copy()

        # ============================================================
        # 自适应联合 cost 权重 (adaptive joint cost weights)
        #
        # 原理:
        #   cost = α·avg_wEPE + β·PVBand + γ·PE
        # 通过一次初始评估，自动将三项归一化到同一量级，
        # 避免人工调参。
        #
        # 归一化:
        #   norm(PVBand) = PVBand / PVBand₀
        #   norm(EPE)    = avg_EPE / avg_EPE₀
        #   norm(PE)     = PE / PE₀

        # ============================================================
        _eval_mask = self.parametric.render_curve(current_cps) + current_SRAFs
        _pv_ref, _ = self.pv_computer.compute(_eval_mask)
        _ai_ref, _wafer_ref, _ = images_simulation_v2(
            mask_spatial=_eval_mask,
            resist_params=self.simulator.params.resist,
            opt_cache=self.simulator.opt_cache,
        )
        _, _epe_ref_vec = caculate_epe(
            _ai_ref, self.simulator.params.resist.threshold,
            self.simulator.sraf.eps, self.simulator.sraf.num_eps,
            self.simulator.sraf.weight_meef, add_weight=False,
        )
        _wepe_ref_total = caculate_wepe(
                self.simulator.sraf.wepe_caculated,
                _epe_ref_vec,
                self.simulator.sraf.num_weps
            )
        print(f"EPE点数{self.simulator.sraf.num_eps}，wEPE点数{self.simulator.sraf.num_weps}")

        # 与 [MEEF_opt] 日志保持同一口径：平均 wEPE = 总加权和 / wEPE 点数
        _avg_wepe_ref = float(_wepe_ref_total) / max(self.simulator.sraf.num_weps, 1)


        # SRAF printing 参考量: 用 extract_sraf 把参考 mask 拆出 SRAF 区域，
        # 然后对 wafer 在该区域内做"灰度平方和"作为 SRAF 印出强度的代理指标。
        _, _sraf_region_ref = extract_sraf(_eval_mask, self.target_mask)
        _sraf_region_ref_bin = (np.asarray(_sraf_region_ref) > 0)
        if _sraf_region_ref_bin.any():
            _sprint_ref = float(np.sum(np.asarray(_wafer_ref)[_sraf_region_ref_bin] ** 2))
        else:
            _sprint_ref = 0.0

        # 目标比例（同量级归一化后再加权）
        # 注: 用 SRAF-printing 平方和代替原 PE 项，更直接惩罚 SRAF 被印出
        w_pv, w_wepe, w_sprint = 0.9, 0.1, 0.4


        alpha_pv = w_pv / max(float(_pv_ref), 1e-12)
        alpha_epe = w_wepe / max(_avg_wepe_ref, 1e-12)
        alpha_sprint = w_sprint / max(_sprint_ref, 1e-12)
        print(f"[adaptive weights] ref avg_wEPE={_avg_wepe_ref:.4g}, "
              f"ref PVBand={_pv_ref:.4g}, ref SRAFprint={_sprint_ref:.4g} "
              f"=> α_pv={alpha_pv:.4g}, α_epe={alpha_epe:.4g}, α_sprint={alpha_sprint:.4g}")


        def cost_compute(width, structure_info, return_details=False):

            """
            计算给定 SRAF 宽度向量下的联合 cost = 0.7·norm(PVBand)+0.1·norm(EPE)+0.2·norm(PE)。
            主图形 cps 在 SRAF 优化阶段保持固定。
            """
            base_cps = structure_info["base_cps"]
            if not self.simulator.sraf.sraf_simplify:
                srafs = distance_msaa_multi_skeleton(
                    structure_info["srafs_skelet_layer"],
                    width
                )
            else:
                srafs = distance_msaa_multi_curves(
                    structure_info["fitted_curves_by_order"],
                    width,
                    mask_shape
                )
            mask = self.parametric.render_curve(base_cps) + srafs

            # PVBand 项: 防止 SRAF 太宽自身被打印 (w>4.0 后急剧爆炸)
            pv_cost, _ = self.pv_computer.compute(mask)

            ai, wafer, _ = images_simulation_v2(
                mask_spatial=mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache,
            )
            _, epe_vec = caculate_epe(
                ai, self.simulator.params.resist.threshold,
                self.simulator.sraf.eps, self.simulator.sraf.num_eps,
                self.simulator.sraf.weight_meef, add_weight=False,
            )
            avg_epe = float(np.mean(np.abs(epe_vec))) if len(epe_vec) > 0 else 0.0
            wepe_total = caculate_wepe(
                self.simulator.sraf.wepe_caculated,
                epe_vec,
                self.simulator.sraf.num_weps,
            )
            avg_wepe = float(wepe_total) / max(self.simulator.sraf.num_weps, 1)

            # SRAF printing 项: 取 SRAF 区域内 wafer 灰度的平方和，
            # 直接对应 "SRAF 是否被印出"，比全图 PE 更有针对性。
            _, sraf_region = extract_sraf(mask, self.target_mask)
            sraf_region_bin = (np.asarray(sraf_region) > 0)
            if sraf_region_bin.any():
                sprint_cost = float(np.sum(np.asarray(wafer)[sraf_region_bin] ** 2))
            else:
                sprint_cost = 0.0

            contrib_pv = alpha_pv * float(pv_cost)
            contrib_epe = alpha_epe * avg_wepe
            contrib_sprint = alpha_sprint * sprint_cost
            joint_cost = contrib_pv + contrib_epe + contrib_sprint

            if return_details:
                return joint_cost, {
                    "pv_cost": float(pv_cost),
                    "avg_epe": float(avg_epe),
                    "avg_wepe": float(avg_wepe),
                    "sprint_cost": float(sprint_cost),
                    "contrib_pv": float(contrib_pv),
                    "contrib_epe": float(contrib_epe),
                    "contrib_sprint": float(contrib_sprint),
                    "joint_cost": float(joint_cost),
                }


            return joint_cost


        # 累计总用时计算
        duration_new = 0  # 在循环外初始化，防止 first iteration 之后引用未绑定
        for iteration_idx in range(1, self.iterations + 1):
            print(f"============= 第 {iteration_idx} 次迭代开始 ==============")
            start_time = time.time()

            # --- 1. 初始化优化配置 ---
            # 根据是否固定 SRAF 决定优化模式
            only_sraf = True if not self.simulator.sraf.fix_sraf else False

            # x0 = np.zeros(dim_x)          # 优化初始向量
            w_min, w_max = 1.0, 4.0       # 定义 SRAF 宽度的物理限制范围

            # 自动选择优化算法 (从 config.yaml 的 sraf.sraf_solver 读取, 默认 CMAES)
            # 可选:
            #   "CMAES"  纯 CMA-ES (默认, 单优化器一步到位, 不接 SLSQP 精化,
            #            起点放空间中心 + σ₀=span/4, 完全不依赖初值)
            #   "DE"     Differential Evolution + polish=True
            #            (内置 L-BFGS-B 抛光, Sobol 初始化, 全局搜索能力最强)
            #   "SLSQP"  拟牛顿 + 多起点重启 (依赖初值, 但局部精度高)
            #   "one_step_GD"  废弃
            if self.simulator.sraf.pvband_cost:
                optimizer = getattr(self.simulator.sraf, "sraf_solver", "CMAES")
            else:
                optimizer = "one_step_GD" # 默认使用单步梯度下降（没有用到）

            if not self.simulator.sraf.sraf_orders:
                optimizer = '1D bounded optimizer' # 如果不分阶，降维为 1D 边界优化

            # 仅在首次迭代时打印配置摘要
            if iteration_idx == 1:
                # pvband_cost=True 时实际是归一化联合损失:
                # 0.7·norm(PVBand) + 0.2·norm(EPE) + 0.1·norm(PE)
                cost_type = (f"joint({w_pv}·PV+{w_wepe}·EPE+{w_sprint}·SRAFprint, α_pv={alpha_pv:g},α_epe={alpha_epe:g},α_sprint={alpha_sprint:g})"


                             if self.simulator.sraf.pvband_cost else "epe")

                def align_text(text, width):
                    # 计算中文字符的数量
                    chinese_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
                    # 实际填充长度 = 目标宽度 - 中文数量（因为每个中文多占1个视觉宽度）
                    return text.ljust(width - chinese_count)

                # --- 打印部分 ---
                print("-" * 50)
                # 使用 align_text 来对齐左侧文字
                print(f"{align_text('配置项目', 25)} | 配置状态")
                print("-" * 50)
                print(f"{align_text('损失函数选择', 25)} : {cost_type}")
                print(f"{align_text('SRAF骨架图采用参数化曲线', 25)} : {self.simulator.sraf.sraf_simplify}")
                print(f"{align_text('是否固定宽度为初始值', 25)} : {self.simulator.sraf.fix_sraf}")
                print(f"{align_text('是否进行分阶优化', 25)} : {self.simulator.sraf.sraf_orders}")
                print(f"{align_text('是否只优化 SRAF', 25)} : {only_sraf}")
                print("-" * 50)
                print("Start optimization...")

            # --- 2. 构建优化边界 (Bounds) ---
            bounds = []
            if not self.simulator.sraf.fix_sraf:
                # 为每个 SRAF 实例分配相同的宽度限制
                bounds = [(w_min, w_max) for _ in range(self.simulator.sraf.num_srafs)]

            # --- 3. 封装当前迭代的结构信息 ---
            structure_info = {
                "base_cps": current_cps,
                "base_vectors": current_vectors,
                "base_SRAFs": self.simulator.sraf.LSM_ordered_srafs,
                "base_sraf_width": current_sraf_width,
                "fix_sraf": self.simulator.sraf.fix_sraf,
                "initial_mask": initial_mask,
                "initial_eps": self.simulator.sraf.eps,
                "num_EPs": self.simulator.sraf.num_eps,
                "delta_cp": self.simulator.sraf.delta,
                "delta_sraf": self.simulator.sraf.delta,
                "curve_type": self.simulator.sraf.curve_type,
                "tr": self.simulator.params.resist.threshold,
                "num_srafs": self.simulator.sraf.num_srafs,
                "target": self.target_mask,
                "num_cps": self.simulator.sraf.num_cps,
                "min_sraf_width": w_min,
                "max_sraf_width": w_max,
                "iteration_idx": iteration_idx,
                "only_sraf": only_sraf,
                "bounds": bounds,
                "sraf_layers": self.simulator.sraf.srafs_layer,
                "srafs_skelet_layer": self.simulator.sraf.srafs_skelet_layer,
                "fitted_curves_by_order":self.simulator.sraf.fitted_curves_by_order
            }

            w_min, w_max = float(w_min), float(w_max)

            # --- 4. 执行优化流程 ---

            # 方案 A: 基于联合 cost (α·wEPE + β·PVBand) 的 SRAF 宽度优化
            #   - PVBand 在 [w_min, w_max] 内对宽度几乎不敏感, 只在 w>4.0 爆炸,
            #     起"防爆护栏"作用;
            #   - wEPE 才是 SRAF 真正的物理价值载体 (诊断脚本验证, 内部最低点
            #     在 w*≈2.6, 改善 99.9%);
            #   - 详见上方 alpha_wepe / beta_pvband 的注释.
            if self.simulator.sraf.pvband_cost:
                num_srafs_opt = self.simulator.sraf.num_srafs

                # ====== 直接参数化 + 排序 (sorted-sample) ======
                # 优化变量 z = [u_1, u_2, ..., u_K], 长度 K = num_srafs_opt
                #   u_i ∈ [w_min, w_max]                      (独立同界)
                # 反解:
                #   w = sort(z, descending)                   保证单调递减
                #
                # 物理保证:
                #   1) ∀i: w[i] ∈ [w_min, w_max] 严格成立    (绝不出 <w_min)
                #   2) 单调: w[0] ≥ w[1] ≥ ... ≥ w[K-1]      (排序天然保证)
                #   3) 允许相等 (cost 鼓励等宽时, 优化器自由选择)
                #
                # 设计取舍:
                #   - 之前累加差量参数化 z=[b,δ1..δ_{K-1}] 在 bounds 常量化时
                #     无法同时保证 w[K-1] ≥ w_min, 必须依赖 np.clip 兜底,
                #     而 clip 会让"不同 z 映射到同一 w"造成 cost landscape 平台,
                #     CMA-ES/DE 在平台上随机游走 → 出现 z 越界但 cost 一样的退化解.
                #   - 直接参数化 + 排序: bounds 全是 [w_min, w_max] 独立同界,
                #     z → w 是连续函数 (排序对优化器友好), 不依赖任何 clip.
                #   - 排序使搜索空间相比"任意 K 维方块"被压缩 K! 倍, 但所有
                #     现代黑盒优化器 (CMA-ES/DE/SLSQP) 都能正常工作,
                #     仅需确保起点在排序后的可行域里 (即起点本身已单调).
                # ================================================
                def z_to_w(z):
                    # 降序排序, 同时强制 clip 到 [w_min, w_max] 防数值漂移
                    w = np.sort(np.asarray(z, dtype=float))[::-1]
                    return np.clip(w, w_min, w_max)

                # 记录优化器内部评估轨迹（主要用于 CMA-ES）
                eval_counter = {"n": 0}
                eval_cost_history = []

                def objective(z):
                    w = z_to_w(z)
                    eval_counter["n"] += 1
                    cost_val, detail = cost_compute(w, structure_info, return_details=True)
                    eval_cost_history.append(float(detail["joint_cost"]))

                    # 每 N 次 objective 调用打印一次（CMA 内部真实迭代轨迹）
                    if optimizer == "CMAES":
                        denom = max(abs(detail["joint_cost"]), 1e-12)
                        print(
                            f"[inner-eval {eval_counter['n']}] w={np.asarray(w).round(3)} "
                            f"cost={detail['joint_cost']:.6f} | "
                            f"PV={detail['contrib_pv']:.6f} ({detail['contrib_pv']/denom*100:.1f}%), "
                            f"EPE={detail['contrib_epe']:.6f} ({detail['contrib_epe']/denom*100:.1f}%), "
                            f"SPRT={detail['contrib_sprint']:.6f} ({detail['contrib_sprint']/denom*100:.1f}%)"
                        )

                    return cost_val


                if optimizer == "CMAES":
                    # ============================================================
                    # 方案 2: 纯 CMA-ES (单优化器, 一步到位, 不接 SLSQP 精化)
                    #
                    # 设计动机:
                    #   - CMA-ES 维护种群 + 自适应协方差, 起点放空间中心,
                    #     σ₀=span/4 第一代就覆盖整个 K 维立方体, 不依赖初值;
                    #   - 提高 maxfevals=200 让 CMA-ES 跑到自然收敛, 末段
                    #     σ 自动收缩, 精度也能压下来, 不需要 SLSQP 抛光.
                    # ============================================================
                    import cma

                    span = w_max - w_min
                    # 起点 = 空间几何中心, K 维全是 (w_min+w_max)/2
                    z_center = np.full(
                        num_srafs_opt, (w_min + w_max) / 2.0, dtype=float,
                    )
                    sigma0 = span / 8.0   # 初始搜索半径覆盖整个空间

                    cma_bounds = [
                        [w_min] * num_srafs_opt,
                        [w_max] * num_srafs_opt,
                    ]
                    # 默认 popsize=4+3*ln(N) 对 4 维约 8, 这里固定 8
                    popsize = 8
                    maxfevals = 200       # 加大预算让 CMA-ES 跑到自然收敛

                    print(f"优化器是：纯 CMA-ES "
                          f"(popsize={popsize}, maxfevals={maxfevals}, "
                          f"w_i∈[{w_min:.2f},{w_max:.2f}], 排序参数化)")
                    try:
                        es = cma.CMAEvolutionStrategy(
                            z_center, sigma0,
                            inopts={
                                "bounds": cma_bounds,
                                "maxfevals": maxfevals,
                                "popsize": popsize,
                                "tolx": 1e-2,    # 收敛容忍度
                                "tolfun": 1e-3,
                                "verbose": -9,   # 关掉 cma 的内部日志
                                "seed": 0,
                            },
                        )
                        es.optimize(lambda z: float(objective(np.asarray(z))))
                        z_opt = np.asarray(es.result.xbest, dtype=float)
                        pvband_opt_result = float(es.result.fbest)
                        n_evals = int(es.result.evaluations)
                        n_iter = int(es.result.iterations)

                        x_opt = z_to_w(z_opt)
                        diffs_opt = -np.diff(x_opt)   # 相邻阶宽度差 (>=0)
                        print(f"  [CMA-ES 收敛] w={x_opt.round(3)}, "
                              f"Δw={diffs_opt.round(3)}, "
                              f"cost={pvband_opt_result:.6f}, "
                              f"评估次数={n_evals}, 代数={n_iter}")
                    except Exception as e:
                        print(f"  [WARN] CMA-ES 失败 ({e}), 退化到当前宽度.")
                        x_opt = current_sraf_width.copy()
                        pvband_opt_result = float(np.inf)

                elif optimizer == "DE":
                    # ============================================================
                    # 方案 1: Differential Evolution (单优化器, 一步到位)
                    #
                    # 设计动机:
                    #   - DE 是 scipy 内置的全局优化器, 种群进化 + 变异/交叉,
                    #     对扁平/多盆地 cost landscape 鲁棒, 不依赖初值;
                    #   - polish=True 让 DE 在收敛后内置一次 L-BFGS-B 精化,
                    #     工程上"一行代码 + 一次调用"就拿到全局 + 局部最优;
                    #   - workers=1: objective 是闭包函数, 多进程无法 pickle;
                    #     单次 cost_compute 较快 (毫秒级), 单进程完全能接受.
                    # ============================================================
                    from scipy.optimize import differential_evolution

                    de_bounds = [(w_min, w_max)] * num_srafs_opt

                    print(f"优化器是：Differential Evolution "
                          f"(polish=True, workers=1, "
                          f"w_i∈[{w_min:.2f},{w_max:.2f}], 排序参数化)")
                    try:
                        de_res = differential_evolution(
                            objective,
                            bounds=de_bounds,
                            strategy="best1bin",
                            popsize=15,           # 种群 = 15 * dim = 60
                            maxiter=80,           # 上限代数
                            tol=1e-7,
                            mutation=(0.5, 1.0),  # 默认抖动
                            recombination=0.7,
                            init="sobol",         # 拟随机初始种群, 覆盖更均匀
                            polish=True,          # 收敛后内置 L-BFGS-B 抛光
                            workers=1,            # 闭包 objective 不可 pickle
                            updating="immediate", # workers=1 用 immediate 收敛更快
                            seed=0,
                        )
                        z_opt = np.asarray(de_res.x, dtype=float)
                        pvband_opt_result = float(de_res.fun)
                        x_opt = z_to_w(z_opt)
                        diffs_opt = -np.diff(x_opt)
                        print(f"  [DE 收敛] w={x_opt.round(3)}, "
                              f"Δw={diffs_opt.round(3)}, "
                              f"cost={pvband_opt_result:.6f}, "
                              f"评估次数={de_res.nfev}, 代数={de_res.nit}, "
                              f"success={de_res.success}")
                    except Exception as e:
                        print(f"  [WARN] DE 失败 ({e}), 退化到当前宽度.")
                        x_opt = current_sraf_width.copy()
                        pvband_opt_result = float(np.inf)

                elif optimizer == "SLSQP":
                    # bounds: w_i ∈ [w_min, w_max], 排序参数化天然保证单调
                    slsqp_bounds = [(w_min, w_max)] * num_srafs_opt
                    span = w_max - w_min

                    # 多初始值重启: 注意起点必须本身就单调递减 (= 已排序),
                    # 否则 SLSQP 在排序的不可微点附近梯度方向乱.
                    z0_candidates = []
                    # 起点 1: 等宽 + 居中
                    z0 = [(w_min + w_max) / 2.0] * num_srafs_opt
                    z0_candidates.append(np.array(z0, dtype=float))
                    # 起点 2: 线性递减覆盖全域
                    z0 = list(np.linspace(w_max, w_min, num_srafs_opt))
                    z0_candidates.append(np.array(z0, dtype=float))
                    # 起点 3: 高位密集递减 (探索"最高阶大其它阶都小")
                    z0 = [w_max] + [w_min + 0.1 * span] * (num_srafs_opt - 1)
                    z0_candidates.append(np.array(z0, dtype=float))

                    best_res = None
                    best_fun = np.inf
                    print(f"优化器是：SLSQP（排序参数化 + 多初始值重启, "
                          f"w_i∈[{w_min:.2f},{w_max:.2f}]）")
                    for z0 in z0_candidates:
                        # 把 z0 clip 到 bounds 内, 防止数值边界异常
                        z0 = np.clip(z0, w_min, w_max)
                        try:
                            res = minimize(
                                objective, z0, method='SLSQP',
                                bounds=slsqp_bounds,
                                options={
                                    'eps': 1e-2,
                                    'maxiter': 100,
                                    'ftol': 1e-6,
                                    'disp': False,
                                }
                            )
                            w_view = z_to_w(res.x)
                            if res.fun < best_fun:
                                best_fun = res.fun
                                best_res = res
                                print(f"  [restart] z0={z0.round(3)} -> "
                                      f"w={w_view.round(3)}, "
                                      f"cost={res.fun:.6f}, "
                                      f"success={res.success}")
                        except Exception as e:
                            print(f"  [restart] z0={z0.round(3)} 失败: {e}")

                    if best_res is None:
                        print("  [WARN] 所有初始值均失败，保持当前宽度。")
                        x_opt = current_sraf_width.copy()
                        pvband_opt_result = float(np.inf)
                    else:
                        x_opt = z_to_w(best_res.x)  # type: ignore[union-attr]
                        pvband_opt_result = float(best_res.fun)  # type: ignore[union-attr]
                        diffs_opt = -np.diff(x_opt)
                        print(f"  [SLSQP 最优] w={x_opt.round(3)}, "
                              f"Δw={diffs_opt.round(3)}, "
                              f"cost={pvband_opt_result:.6f}")

                # 如果不分阶，使用 1D 标量优化
                if optimizer == "1D bounded optimizer":
                    print(">>> Using 1D bounded optimizer")
                    result = minimize_scalar(
                        lambda w: cost_compute(np.array([w]), structure_info),
                        bounds=(w_min, w_max),
                        method="bounded",
                        options={"xatol": 1e-4, "maxiter": 500}
                    )
                    x_opt = np.array([result.x])
                    pvband_opt_result = float(result.fun)

                # 导出求解过程每次评估的 cost 为 CSV，用于画损失曲线
                if len(eval_cost_history) > 0:
                    trace_dir = Path(self.simulator.sraf.filepath)
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    cost_df = pd.DataFrame({
                        "eval_id": np.arange(1, len(eval_cost_history) + 1),
                        "cost": eval_cost_history,
                    })
                    cost_path = trace_dir / f"cost_curve_iter_{iteration_idx}.csv"
                    cost_latest_path = trace_dir / "cost_curve_latest.csv"
                    cost_df.to_csv(cost_path, index=False, encoding="utf-8-sig")
                    cost_df.to_csv(cost_latest_path, index=False, encoding="utf-8-sig")
                    print(f"[trace] 求解过程 cost 已保存: {cost_path}")

                # 更新 SRAF 宽度并重新计算图形结构
                current_sraf_width = x_opt


                if not self.simulator.sraf.sraf_simplify:
                    current_SRAFs = distance_msaa_multi_skeleton(
                        structure_info["srafs_skelet_layer"],
                        current_sraf_width
                    )
                else:
                    current_SRAFs = distance_msaa_multi_curves(
                        structure_info["fitted_curves_by_order"],
                        current_sraf_width,
                        mask_shape
                    )

            # --- 5. 统计与打印迭代结果 ---
            end_time = time.time()
            iter_duration = end_time - start_time
            print(f"第 {iteration_idx} 次迭代完成，用时 {iter_duration:.2f}s")

            # 累计总用时计算
            duration_new = int(iter_duration + duration_new)

            print("当前 SRAF 宽度:", ", ".join(f"{w:.3f}" for w in current_sraf_width))

            # ================== 1. 当前掩模成像与仿真 ==================
            # 渲染当前曲线并叠加 SRAF 得到最终掩模
            current_mask = self.parametric.render_curve(current_cps) + current_SRAFs

            # 进行图像仿真 (光学强度 AI, 光刻胶图形 Wafer)
            ai_current, wafer_current, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )

            # ================== 2. 误差指标计算 ==================
            # 计算 EPE (Edge Placement Error)
            current_epe, current_epe_vector = caculate_epe(
                ai_current,
                self.simulator.params.resist.threshold,
                self.simulator.sraf.eps,
                self.simulator.sraf.num_eps,
                self.simulator.sraf.weight_meef,
                add_weight=False
            )

            # 计算 wEPE (Weighted EPE)
            current_wepe = caculate_wepe(
                self.simulator.sraf.wepe_caculated,
                current_epe_vector,
                self.simulator.sraf.num_weps
            )

            # 计算 PE (Pixel Error)
            current_pe = np.round(np.sum((self.target_mask - wafer_current) ** 2), 6)

            # 打印当前迭代的各项指标
            avg_wepe_current = current_wepe / max(self.simulator.sraf.num_weps, 1)
            avg_epe_current = current_epe / max(self.simulator.sraf.num_eps, 1)
            print(f"迭代 {iteration_idx} 次后：平均wEPE = {avg_wepe_current:.6f}，"
                  f"EPE = {avg_epe_current:.6f}，PE = {current_pe:.6f}")

            # 计算 PVBand (Process Variation Band)

            # 注意: 联合 cost 模式下 pvband_opt_result 是 α·wEPE+β·PVBand 的
            # 加权和, 不是纯 PVBand, 这里需要重新对最优 mask 计算一次纯 PVBand
            # 用于报告 (开销 ~几秒, 一次迭代仅一次).
            if self.simulator.sraf.pvband_cost:
                current_pvband, _ = self.pv_computer.compute(current_mask)
                current_pvband = float(current_pvband)

                # 打印每轮联合损失三项贡献（与当前 cost_compute 实现口径一致）
                contrib_pv = alpha_pv * current_pvband
                contrib_epe = alpha_epe * avg_wepe_current

                # SRAF printing 项：当前 mask 的 SRAF 区域内 wafer 灰度平方和
                _, _sraf_region_cur = extract_sraf(current_mask, self.target_mask)
                _sraf_region_cur_bin = (np.asarray(_sraf_region_cur) > 0)
                if _sraf_region_cur_bin.any():
                    current_sprint = float(
                        np.sum(np.asarray(wafer_current)[_sraf_region_cur_bin] ** 2)
                    )
                else:
                    current_sprint = 0.0
                contrib_sprint = alpha_sprint * current_sprint

                joint_cost = contrib_pv + contrib_epe + contrib_sprint
                denom = max(abs(joint_cost), 1e-12)
                print(
                    f"迭代 {iteration_idx} 次后：PV Loss = {current_pvband:.6f}  "
                    f"SRAFprint = {current_sprint:.6f}  "
                    f"(联合 cost = {joint_cost:.6f})"
                )
                print(
                    f"[contrib] PV={contrib_pv:.6f} ({contrib_pv/denom*100:.1f}%), "
                    f"EPE={contrib_epe:.6f} ({contrib_epe/denom*100:.1f}%), "
                    f"SPRT={contrib_sprint:.6f} ({contrib_sprint/denom*100:.1f}%)"
                )
            else:
                current_pvband = 0

            # ================== 3. 结果保存与历史记录 ==================

            save_width_results(
                self.simulator.sraf.filepath, current_mask, current_SRAFs,
                ai_current, wafer_current, current_sraf_width, 'cividis',
                pixel_size=self.simulator.mask.pixel_size
            )

            # 记录历史数据用于后续绘图
            epe_errors.append(current_epe / self.simulator.sraf.num_eps)
            pe_errors.append(current_pe)
            iterations.append(iteration_idx)
            wepe_errors.append(current_wepe / self.simulator.sraf.num_weps)
            iteration_time.append(duration_new)
            pvband_errors.append(current_pvband)

            # --- 更新 wEPE 最优解 ---
            if current_wepe < best_wepe:
                best_wepe = current_wepe
                best_wepe_epe = current_epe
                best_wepe_mask = current_mask.copy()
                best_wepe_wafer = wafer_current.copy()
                best_wepe_it = iteration_idx
                best_wepe_cps = current_cps.copy()
                best_wepe_pvband = current_pvband

            # --- 更新 EPE 最优解 ---
            if current_epe < best_epe:
                best_epe = current_epe
                best_epe_wepe = current_wepe
                best_epe_mask = current_mask.copy()
                best_epe_wafer = wafer_current.copy()
                best_epe_it = iteration_idx
                best_epe_cps = current_cps.copy()
                best_epe_pvband = current_pvband

            # 增量保存 Excel 数据
            save_excel(
                self.simulator.sraf.up_filename,
                self.simulator.sraf.second_filename,
                self.simulator.sraf.curve_type,
                self.simulator.sraf.filename,
                iterations, pe_errors, epe_errors, wepe_errors, iteration_time,
                file_path=self.simulator.sraf.filepath,
            )


        # ================== 4. 优化结束：持久化最优结果 ==================
        save_dir = Path(self.simulator.sraf.filepath)

        # --- A. WEPE 最优结果持久化 ---
        save_path_wepe = save_dir / "best_wepe_result.txt"
        with open(save_path_wepe, "w", encoding="utf-8") as f:
            f.write(f"迭代过程中WEPE最优为第 {best_wepe_it} 次迭代\n"
                    f"EPE误差为 {best_wepe_epe / self.simulator.sraf.num_eps}\n"
                    f"WEPE误差为 {best_wepe / self.simulator.sraf.num_weps}\n"
                    f"PVband为 {best_wepe_pvband}\n")

        with open(save_dir / "WEPE最优的控制点坐标.txt", "w") as f:
            for row in best_wepe_cps:
                f.write(" ".join(map(str, row)) + "\n")

        highlight_contour_red(self.target_mask, best_wepe_wafer,
                              save_path=f"{save_dir}/优化后WEPE最优wafer.png")
        plt.imsave(f"{save_dir}/优化后WEPE最优mask.png", best_wepe_mask, cmap="cividis")

        # --- B. EPE 最优结果持久化 ---
        save_path_epe = save_dir / "best_epe_result.txt"
        with open(save_path_epe, "w", encoding="utf-8") as f:
            f.write(f"迭代过程中EPE最优为第 {best_epe_it} 次迭代\n"
                    f"EPE误差为 {best_epe / self.simulator.sraf.num_eps}\n"
                    f"WEPE误差为 {best_epe_wepe / self.simulator.sraf.num_weps}\n"
                    f"PVband为 {best_epe_pvband}\n")

        with open(save_dir / f"EPE最优第{best_epe_it}次的控制点坐标.txt", "w") as f:
            for row in best_epe_cps:
                f.write(" ".join(map(str, row)) + "\n")

        highlight_contour_red(self.target_mask, best_epe_wafer,
                              save_path=f"{save_dir}/优化后EPE最优wafer.png")
        plt.imsave(f"{save_dir}/优化后EPE最优mask.png", best_epe_mask, cmap="cividis")

        # ================== 5. PVBand 性能对比统计 ==================
        pv_loss_LSM, _ = self.pv_computer.compute(self.simulator.sraf.LSM_mask)
        pv_loss_ordered_LSM, _ = self.pv_computer.compute(self.simulator.sraf.LSM_initial_mask)
        pv_loss_opt, _ = self.pv_computer.compute(best_wepe_mask)
        pv_loss_target, _ = self.pv_computer.compute(self.target_mask)

        save_path_pv = save_dir / "pvband_loss.txt"
        with open(save_path_pv, "w", encoding="utf-8") as f:
            summary_info = (
                f"目标mask的PV Loss: {pv_loss_target:.6f}\n"
                f"LSM的mask的PV Loss: {pv_loss_LSM:.6f}\n"
                f"分阶初始化宽度的mask的PV Loss: {pv_loss_ordered_LSM:.6f}\n"
            )
            if not self.simulator.sraf.pvband_cost:
                summary_info += f"main优化的mask的PV Loss: {pv_loss_opt:.6f}"
            else:
                summary_info += (f"mianEPE优化后的mask的PV Loss: {pvband_opt_main:.6f}\n"
                                f"sraf优化后的mask的PV Loss: {pv_loss_opt:.6f}")
            f.write(summary_info)
            print(summary_info)

        # ================== 6. 绘制收敛曲线 ==================
        error_path = f"{self.simulator.sraf.filepath}/error"
        drawcostcurve(epe_errors, iterations, f'{error_path}/EPE_error.png', 'EPE_error', 'EPE')
        drawcostcurve(wepe_errors, iterations, f'{error_path}/wEPE_error.png', 'wEPE_error', 'wEPE')
        drawcostcurve(pe_errors, iterations, f'{error_path}/PE_error.png', 'PE_error', 'PE')
        drawcostcurve(pvband_errors, iterations, f'{error_path}/pvband_error.png', 'pvband_error', 'pvband')
