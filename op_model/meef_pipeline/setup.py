"""MEEFPipelineSetup: 装配出与 set_meef.MEEF 鸭子兼容的对象.

使命:
    把 LSM 优化后的版图 + yaml 配置, 一次性变成 MEEF_Optimizer 可用的
    "simulator.meef" 替身. 暴露的字段与 set_meef.MEEF 一致, 便于:
      simulator.meef = MEEFPipelineSetup(...)
      MEEF_Optimizer(simulator).run()  # 不改一行优化器代码

字段:
    curve_type, pattern_name, file_name
    up_filename, second_filename, filename, filepath
    cps, num_cps                              # 主图形 CP (本管线唯一优化变量)
    eps, num_eps, weight_meef, wepe_caculated, num_weps  # EP 点
    delta                                     # MEEF 差分扰动量
    initial_mask                              # 主图形+SRAF 的 CP 渲染版 (起点)
    sraf_mask                                 # SRAF 由 CP+B-spline+MSAA 渲染的常量

新增字段 (相比 set_meef.MEEF):
    sraf_cps, num_sraf_cps                    # 留作可视化, 优化期间不动
    lsm_mask, main_mask                       # 输入 LSM 与主图形分量
    pipeline_meta                             # 配置参数快照 (debug 用)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import numpy as np

from utils_model.demo_MSAA import AntiAliasRenderer
from utils_model.demo_parametric import ParametricDemo
from utils_model.project_paths import OPC_OUTPUT_DIR

from .mask_loader import load_lsm_mask, split_main_and_sraf, align_to_target_shape
from .cp_extractor import extract_main_cps, extract_sraf_cps
from .sraf_renderer import render_sraf_mask


@dataclass
class MEEFPipelineConfig:
    """yaml 中 meef_pipeline 段的解析结果."""

    # 必填
    pattern_name: str
    ls_mask_path: str
    file_name: str
    move_strategy: str
    # 主图形 CP
    main_cp_interval: int = 5
    main_symmetry: str = "none"            # none / left-right / diagonal / center
    # SRAF CP
    sraf_cp_arclen: float = 3.0            # 沿弧长每隔多少像素一个 CP
    sraf_min_cps: int = 8
    sraf_smoothing: float = 0.3
    sraf_curve_pts: int = 200
    sraf_min_area: int = 20
    # 渲染
    msaa_level: int = 16
    # 优化器需要的标量
    curve_type: str = "BS"
    delta: float = 0.15
    # 拆分
    dilate_radius: int = 2
    # EP 点 (其它非通孔图形用)
    interval_line: int = 5
    interval_corner: int = 2
    mid_weight: float = 4.0
    other_weight: float = 1.0


class MEEFPipelineSetup:
    """与 set_meef.MEEF 鸭子兼容的装配类."""

    def __init__(self, target_mask: np.ndarray, cfg: MEEFPipelineConfig):
        self.cfg = cfg

        # ---------- 0. 标量字段 (与 set_meef.MEEF 同名) ----------
        self.target_mask = np.asarray(target_mask).copy()
        self.curve_type = cfg.curve_type
        self.pattern_name = cfg.pattern_name
        self.file_name = cfg.file_name
        self.delta = float(cfg.delta)
        # 控制点移动策略 (与 set_meef.MEEF 同名), 供 MEEF_Optimizer 读取:
        # "bisector"(角平分线一维位移) / "xy"(X/Y 解耦合成二维位移)
        self.move_strategy = getattr(cfg, "move_strategy", "bisector")

        # ---------- 1. 输出目录 (统一放在 outputs/opc) ----------
        self.up_filename = str(OPC_OUTPUT_DIR)
        self.second_filename = "MEEF_pipeline"
        self.filename = (
            f"{cfg.pattern_name}_"
            f"k={cfg.main_cp_interval}_sym={cfg.main_symmetry}_"
            f"srafArc={cfg.sraf_cp_arclen}_{cfg.curve_type}_"
            f"{cfg.file_name}"
        )
        out_dir = Path(self.up_filename) / self.second_filename / cfg.curve_type / self.filename
        out_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = str(out_dir).replace("\\", "/")

        # ---------- 2. 加载 LSM, 拆分 main/SRAF ----------
        self.lsm_mask = load_lsm_mask(cfg.ls_mask_path)
        if self.lsm_mask.shape != self.target_mask.shape:
            print(
                f"[MEEFPipeline] LSM shape {self.lsm_mask.shape} 与 target "
                f"{self.target_mask.shape} 不一致, 自动 center-crop / pad 对齐."
            )
            self.lsm_mask = align_to_target_shape(self.lsm_mask, self.target_mask.shape)
        self.main_mask, sraf_mask_raw = split_main_and_sraf(
            self.lsm_mask, self.target_mask, dilate_radius=cfg.dilate_radius,
        )

        # ---------- 3. 主图形 CP + EP 点 ----------
        self.cps, eps_helper = extract_main_cps(
            self.target_mask,
            interval_k=cfg.main_cp_interval,
            symmetry=cfg.main_symmetry,
            pattern_name=cfg.pattern_name,
            mid_weight=cfg.mid_weight,
            other_weight=cfg.other_weight,
        )
        self.num_cps = sum(len(c) for c in self.cps)

        # EP 点选取规则: 通孔类 -> _select_eps_ofvia, 否则 -> _select_eps_ofothers
        # 与 set_meef.MEEF._select_ep_points 完全一致, 自动判断
        if cfg.pattern_name in ("中心对称通孔", "对角通孔"):
            eps, wepe_caculated, weight_meef = eps_helper._select_eps_ofvia(r=2)
        else:
            eps, wepe_caculated, weight_meef = eps_helper._select_eps_ofothers(
                cfg.interval_line, cfg.interval_corner,
            )
            eps_helper.debug_plot_eps(eps, wepe_caculated, weight_meef, self.filepath, show=False)
        self.eps = np.asarray(eps).reshape(-1, 2)
        self.wepe_caculated = wepe_caculated
        self.weight_meef = weight_meef
        self.num_eps = len(self.eps)
        self.num_weps = int(np.sum(np.array(self.wepe_caculated) == 1))

        # ---------- 4. SRAF CP -> 闭合 B-spline -> MSAA 灰度图 (常量) ----------
        # sraf_cps:             每个连通块的一组闭合 CP (用于优化期 forward 重渲染)
        # sraf_contour_points:  所有 SRAF 原始外轮廓像素点总数 (调试/统计用)
        self.sraf_cps: List[np.ndarray]
        self.sraf_contour_points: int
        self.sraf_cps, self.sraf_contour_points = extract_sraf_cps(
            sraf_mask_raw,
            arclen_step=cfg.sraf_cp_arclen,
            min_cps=cfg.sraf_min_cps,
            symmetry=cfg.main_symmetry,
            min_block_area=cfg.sraf_min_area,
        )
        self.num_sraf_cps = sum(len(c) for c in self.sraf_cps)

        self.sraf_mask = render_sraf_mask(
            self.sraf_cps,
            mask_shape=self.target_mask.shape,
            smoothing=cfg.sraf_smoothing,
            num_points=cfg.sraf_curve_pts,
            msaa_level=cfg.msaa_level,
        )

        # ---------- 5. 主图形 CP 拟合 + 与 sraf_mask 拼接 = initial_mask ----------
        # 与 MEEF_Optimizer 内的 forward 链一致:
        #   current_mask = parametric.render_curve(cps) + sraf_mask
        # 所以这里同样用 ParametricDemo 渲染主图形, 让 initial_mask 已是 "CP 表示" 下
        # 的起点 (避免 LSM 像素直接当起点带来的不一致).
        parametric = ParametricDemo(self.curve_type, self.target_mask)
        main_gray = parametric.render_curve(self.cps)
        self.initial_mask = main_gray + self.sraf_mask

        # ---------- 6. 元数据 / 持久化辅助 ----------
        self.pipeline_meta = asdict(cfg)
        self._save_meta()

    # ------------------------------------------------------------------
    # 内部: 把 cps / sraf_cps / eps / config 落盘, 方便 debug
    # ------------------------------------------------------------------
    def _save_meta(self):
        out = Path(self.filepath)
        out.mkdir(parents=True, exist_ok=True)
        # 主图形 CP
        with open(out / "main_cps.txt", "w", encoding="utf-8") as f:
            for c_idx, contour in enumerate(self.cps):
                f.write(f"# main contour {c_idx}\n")
                for pt in contour:
                    f.write(f"{float(pt[0]):.4f} {float(pt[1]):.4f}\n")
        # SRAF CP
        with open(out / "sraf_cps.txt", "w", encoding="utf-8") as f:
            for c_idx, contour in enumerate(self.sraf_cps):
                f.write(f"# sraf block {c_idx}\n")
                for pt in contour:
                    f.write(f"{float(pt[0]):.4f} {float(pt[1]):.4f}\n")
        # EP
        np.savetxt(out / "eps.txt", np.asarray(self.eps), fmt="%d")
        # SRAF mask
        np.savetxt(out / "sraf_mask.txt", self.sraf_mask, fmt="%.6f")
        # config dump
        try:
            import json
            with open(out / "pipeline_config.json", "w", encoding="utf-8") as f:
                json.dump(self.pipeline_meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
