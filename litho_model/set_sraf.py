import numpy as np
from pathlib import Path
from skimage import draw, measure
from utils_model.project_paths import OPC_OUTPUT_DIR, case_1_result_path, project_path
from scipy.ndimage import  binary_dilation
from utils_model.plot_matrix import show_matrices
from utils_model.demo_sraf_utils import distance_msaa_multi_curves,visualize_sraf_and_controls,get_b_spline_points_by_order,debug_plot_fitted_curves,classify_sraf_and_get_cps,distance_msaa_multi_skeleton,classify_sraf_byDistance,visualize_srafs_by_order,extract_sraf
from matplotlib import pyplot as plt
from skimage.morphology import dilation, disk, skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import remove_small_objects
from utils_model.select_eps import DemoSelectEps_new

from skimage.morphology import binary_opening, disk
import cv2 
class SRAF:
    def __init__(self,
                 target_mask: np.ndarray,
                 curve_type: str,
                 pattern_name: str,
                 file_name: str,
                 sraf_orders:bool,
                 fix_sraf: bool,
                 epe_cost:bool,
                 pvband_cost:bool,
                 sraf_simplify:bool,
                 meef_opt_cps_path: str = ""):
        """
        meef_opt_cps_path : MEEF 优化后主图形控制点坐标的 TXT 文件路径。
            pvband_cost=True 时必须提供，否则运行时会报 FileNotFoundError。
            支持正斜杠和反斜杠（内部统一转为 pathlib.Path）。
        """
        # --- 1. 基础参数初始化 ---
        self.target_mask = target_mask      # 目标设计图 (Target Layout)
        self.curve_type = curve_type
        self.pattern_name = pattern_name
        self.file_name = file_name
        self.mask_shape = target_mask.shape
        # MEEF 优化后主图形 cps 路径 (pvband_cost=True 时使用)
        self.meef_opt_cps_path = str(project_path(meef_opt_cps_path)) if meef_opt_cps_path else ""
        
        # 优化策略开关
        self.sraf_orders = sraf_orders      # 是否对 SRAF 进行分阶 (1st, 2nd, etc.)
        self.fix_sraf = fix_sraf            # 是否冻结 SRAF 图形
        self.epe_cost = epe_cost            # 主图形优化：使用 EPE (Edge Placement Error)
        self.pvband_cost = pvband_cost      # SRAF 优化：使用 PVBand (Process Variation Band)
        self.sraf_simplify = sraf_simplify  # 是否使用参数化/简化后的 SRAF
        
        # 权重与步长设置（暂时没有用到！！！！）
        self.mid_weight = 10                # 关键控制点权重
        self.other_weight = 1               # 普通采样点权重
        self.delta = 0.15 if self.pvband_cost else 0.01  # 梯度下降步长 delta
        
        # 工艺约束参数
        self.initial_width = 2.5          # SRAF 初始化宽度 (像素)
        # 这两个参数暂时没用到
        self.interval_line = 6              # 直线段采样间隔
        self.interval_corner = 1            # 拐角处采样间隔
        self.lsmispe_epe_pvband = True      # LSM PE EPE/PVBand 优化后的mask
        
        # --- 2. 外部类与文件路径加载 ---
        # 选择EP点的类
        self.select_epsclass = DemoSelectEps_new(
            self.target_mask, self.mid_weight, self.other_weight, self.pattern_name
        )
        # 获取保存文件的路径
        self.filepath = self._get_file()
        
        # 加载 LSM 优化后的初始掩模
        self.LSM_mask, _ = self._load_srafsandinitial_mask()
        ##检查用的
        # plt.imshow(self.LSM_mask)
        # plt.title('LSM_mask')
        # plt.show()
        
        # --- 3. 图形分离与 SRAF 分阶 (Classification) ---
        # 3.1 提取 LSM 掩模中的主图形和 SRAF
        self.LSM_mask_main, self.LSM_SRAF = extract_sraf(self.LSM_mask, self.target_mask)
        
        # 3.2 对 SRAF 按距离主图形的远近进行分阶 (例如 4 阶)，对角9.0，中心对称8.0
        if self.sraf_orders:
            # ---------- 分阶路径 (ordered, 原有逻辑) ----------
            if pattern_name == '中心对称通孔':
                dist_tol1 = 8.0
            elif pattern_name == '对角通孔':
                dist_tol1 = 9.0
            else:
                dist_tol1 = 8.0  # 非通孔类图案默认

            # srafs_layer: 每一阶 SRAF 的独立列表
            # LSM_ordered_srafs:  LSM中所有阶数的 SRAF 
            # srafs_skelet_layer： 每一阶 SRAF 的骨架图
            # LSM_ordered_skelet_srafs： LSM中所有阶数的 SRAF 的骨架图
            # num_pointsOfskelet： 每一阶 SRAF 的骨架图中的点数
            self.srafs_layer, self.LSM_ordered_srafs,self.srafs_skelet_layer,self.LSM_ordered_skelet_srafs,self.num_pointsOfskelet,_ = classify_sraf_byDistance(
                    self.target_mask, self.LSM_SRAF, self.filepath,num_orders=4, dist_tol=dist_tol1
                ) 
            # 可视化分阶结果
            show_matrices([self.srafs_skelet_layer[1],self.srafs_skelet_layer[2],self.srafs_skelet_layer[3],self.srafs_skelet_layer[4]],
                          ["SRAFs by Order 1", "SRAFs by Order 2", "SRAFs by Order 3", "SRAFs by Order 4"],save_path=self.filepath,filename="srafs_by_order_skeletion.png")
            
            # --- 4. 骨架化与拓扑分析 (Skeletonization) ---
            self.skel = skeletonize(self.LSM_ordered_srafs)
            # 计算距离场 (Distance Transform): 用于后续基于中心线恢复 SRAF 宽度
            self.distance_map = distance_transform_edt(~self.skel)
            
            # 对 SRAF 骨架线参数化曲线拟合 (分阶)
            self.sraf_cps_by_order, debug_info = classify_sraf_and_get_cps(self.target_mask,self.LSM_SRAF,num_orders=4, dist_tol=dist_tol1)
            self.num_sraf_cps = debug_info["num_cps"] 
            self.LSM_ordered_srafs = debug_info["all_order_sraf"]
            self.srafs_layer = debug_info["sraf_mask_by_order"]
            self.fitted_curves_by_order = get_b_spline_points_by_order(self.sraf_cps_by_order)
            
            # 可视化输出
            visualize_sraf_and_controls(debug_info["all_order_sraf"],self.sraf_cps_by_order,self.fitted_curves_by_order,self.filepath,self.num_sraf_cps,self.num_pointsOfskelet)
            self.ordered_map = visualize_srafs_by_order(self.srafs_layer, self.filepath)

        else:
            # ---------- 不分阶路径 (flat) ----------
            # 规则：
            # 1) 中心对称通孔/对角通孔：仍做分阶用于筛掉冗余 SRAF，但宽度参数保持单一。
            # 2) 其他图案：保持原始不分阶单层处理。
            from utils_model.demo_sraf_utils import _batch_extract_cps

            via_like_pattern = (self.pattern_name == '中心对称通孔' or self.pattern_name == '对角通孔')

            if via_like_pattern:
                dist_tol1 = 8.0 if self.pattern_name == '中心对称通孔' else 9.0

                # 先按距离分阶，保留有效阶（用于去除多余 SRAF）
                self.srafs_layer, self.LSM_ordered_srafs, self.srafs_skelet_layer, self.LSM_ordered_skelet_srafs, self.num_pointsOfskelet, _ = classify_sraf_byDistance(
                    self.target_mask, self.LSM_SRAF, self.filepath, num_orders=4, dist_tol=dist_tol1
                )

                # 分阶控制点/曲线（仅结构分阶；后续宽度仍是统一单变量）
                self.sraf_cps_by_order, debug_info = classify_sraf_and_get_cps(
                    self.target_mask, self.LSM_SRAF, num_orders=4, dist_tol=dist_tol1
                )
                self.num_sraf_cps = debug_info["num_cps"]
                self.LSM_ordered_srafs = debug_info["all_order_sraf"]
                self.srafs_layer = debug_info["sraf_mask_by_order"]
                self.fitted_curves_by_order = get_b_spline_points_by_order(self.sraf_cps_by_order)

                self.skel = skeletonize(self.LSM_ordered_srafs)
                self.distance_map = distance_transform_edt(~self.skel)
                self.ordered_map = visualize_srafs_by_order(self.srafs_layer, self.filepath, title="SRAFs (flat-width with order)")
                visualize_sraf_and_controls(
                    self.LSM_ordered_srafs, self.sraf_cps_by_order, self.fitted_curves_by_order,
                    self.filepath, self.num_sraf_cps, self.num_pointsOfskelet,
                    title="SRAF Control Points & B-Spline Curve (flat-width with order)",
                )
            else:
                # 其他图案保持单层不分阶
                self.srafs_layer = {0: self.LSM_SRAF}
                self.LSM_ordered_srafs = self.LSM_SRAF

                sraf_bool = self.LSM_SRAF.astype(bool)
                sraf_opened = binary_opening(sraf_bool, disk(2))
                skelet = skeletonize(sraf_bool)

                self.srafs_skelet_layer = {0: skelet.astype(np.uint8)}
                self.LSM_ordered_skelet_srafs = skelet.astype(np.uint8)
                self.num_pointsOfskelet = {0: int(skelet.sum())}

                # CP 提取: 连通域标注 → 批量提取控制点 (不分阶)
                labeled_sraf, num_sraf = label(sraf_opened, connectivity=2, return_num=True)
                if num_sraf > 0:
                    all_labels = [(lb, 0) for lb in range(1, num_sraf + 1)]
                    cps_list, num_cp = _batch_extract_cps(all_labels, labeled_sraf, skelet, spacing=10)
                else:
                    cps_list, num_cp = [], 0
                self.sraf_cps_by_order = {0: cps_list}
                self.num_sraf_cps = num_cp

                self.fitted_curves_by_order = get_b_spline_points_by_order(self.sraf_cps_by_order)

                self.skel = skelet
                self.distance_map = distance_transform_edt(~skelet)
                self.ordered_map = visualize_srafs_by_order({0: self.LSM_SRAF}, self.filepath, title="SRAFs (no-order)")
                visualize_sraf_and_controls(
                    self.LSM_SRAF, self.sraf_cps_by_order, self.fitted_curves_by_order,
                    self.filepath, self.num_sraf_cps, self.num_pointsOfskelet,
                    title="SRAF Control Points & B-Spline Curve (no-order)",
                )
        
        
        # plt.imshow(self.LSM_ordered_skelet_srafs)
        # plt.title('LSM_ordered_skelet_srafs')
        # plt.show()

        # --- 5. 数据持久化 (导出 TXT 检查) ---
        # 写到结果目录, 跟其他 SRAF 优化产物 (mask 图、误差曲线) 放在一起;
        # 同时在原 \"宽度初始化后的LSMmask\" 目录留一份兼容副本, 供 set_meef.py
        # 等下游模块继续按老路径读取 (避免改一处坏一片).
        legacy_dir = OPC_OUTPUT_DIR / "sraf_width_opt" / "宽度初始化后的LSMmask"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(self.filepath)
        result_dir.mkdir(parents=True, exist_ok=True)

        srafs_only = self.LSM_ordered_srafs
        srafs_with_main = self.LSM_ordered_srafs + self.LSM_mask_main

        for out_dir in (result_dir, legacy_dir):
            np.savetxt(out_dir / f"{self.file_name}分阶后无主图形.txt",
                       srafs_only, fmt='%d')
            np.savetxt(out_dir / f"{self.file_name}分阶后有主图形.txt",
                       srafs_with_main, fmt='%d')


        #self.srafs提供sraf的位置
        
        # --- 6. SRAF 宽度初始化与重构 ---
        self.num_srafs = len(self.srafs_layer) if self.sraf_orders else 1
        self.original_width = np.full(self.num_srafs, self.initial_width)

        # 基于分阶信息和中心线，生成给定初始宽度的规则 SRAF (Multi-sampling Anti-Aliasing)
        # 两种形式，根据骨架图或者是根据参数化的骨架图
        # self.srafsin_iniwidth = distance_msaa_multi_sraf(
        #     self.LSM_ordered_srafs, self.original_width, self.srafs_layer, samples=4, show=False
        # )
        if not self.sraf_simplify:
            self.srafsin_iniwidth = distance_msaa_multi_skeleton(
            self.srafs_skelet_layer,self.original_width, samples=4, show=False
        )
        else:
            self.srafsin_iniwidth = distance_msaa_multi_curves(self.fitted_curves_by_order,self.original_width,self.mask_shape)
        # show_matrices([gray_maskOfparametric_curves,self.srafsin_iniwidth], ["gray_maskOfparametric_curves","srafsin_iniwidth"])
        show_matrices([self.srafsin_iniwidth], ["srafsin_iniwidth"],save_path=self.filepath,filename="初始化宽度后的SRAF")
        # 最终的 LSM 初始掩模 = 分离出的主图形 + 重新定义宽度的 SRAF
        self.LSM_initial_mask = self.srafsin_iniwidth + self.LSM_mask_main
        # 同样: 结果目录 + 兼容目录 各写一份
        for out_dir in (result_dir, legacy_dir):
            np.savetxt(out_dir / f"{self.file_name}分阶后有主图形初始化宽度.txt",
                       self.LSM_initial_mask, fmt='%f')
        
        # 7 部分没有用到可以不用管
        # --- 7. 控制点 (CP) 与评估点 (EP) 选取 ---
        # 7.1 选择 EP 点 (用于计算 Cost Function)
        self.eps, self.wepe_caculated, self.weight_meef = self._select_ep_points()
        
        # 7.2 选择 CP 点 (用于参数化控制图形形变)
        self._select_cps()
        # 清洗过近的控制点，防止优化震荡
        self.cps = self.select_epsclass.remove_adjacent_close_points(self.cps, 3)
        
        # 统计点位数量
        self.num_cps = sum(len(v) for v in self.cps)
        self.num_eps = len(self.eps)
        self.num_weps = np.sum(np.array(self.wepe_caculated) == 1)
        
        # # --- 8. 调试可视化 ---
        show_matrices([self.LSM_mask, self.LSM_ordered_srafs + self.LSM_mask_main,self.srafsin_iniwidth + self.LSM_mask_main, self.skel,self.distance_map], 
                      [ "LSM_mask",'LSM_ordered_SRAFs and main', 'main and initial width SRAF','skeleton map','distance_map'],save_path=self.filepath,filename="LSM+分阶后的SRAF+初始化宽度的SRAF+骨架图+距离场")
        
         
        
    
    
    
    
    def _select_cps(self)-> np.ndarray:
        """依据不同的版图样式选取不同的控制点，目前没有用到

        Returns:
            np.ndarray: 控制点坐标
        """
        if self.pattern_name == '中心对称通孔':
            self.cps = self.select_epsclass.extract_mask_control_points(5,symmetry="left-right")
        elif self.pattern_name == '对角通孔':
            self.cps = self.select_epsclass.extract_mask_control_points(5,symmetry="diagonal")
        else:
            self.cps = self.select_epsclass.extract_mask_control_points(5,symmetry=None)   #无对称

    def _select_ep_points(self)-> np.ndarray:
        """
        选取ep点，目前没有用到
        """
        if self.pattern_name == '中心对称通孔' or self.pattern_name == '对角通孔':
            eps,wepe_caculated, weight_meef = self.select_epsclass._select_eps_ofvia(r = 2)
        else:
            eps,wepe_caculated, weight_meef = self.select_epsclass._select_eps_ofothers(self.interval_line, self.interval_corner)                
        return eps,wepe_caculated, weight_meef
    
    
    def _get_dir(self,up_filename,sc_filename,curve_type,filename):
        """
        创建文件目录
        
        """
        base_dir = Path(up_filename) / sc_filename/curve_type/filename
        base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file(self):
        """
        创建保存结果的文件的名称
        """
        self.up_filename = str(OPC_OUTPUT_DIR)
        self.second_filename = 'sraf_width_opt'
        if self.sraf_simplify == False:
            self.filename = (
                            f"宽度={self.initial_width}_"
                            f"分阶={'是' if self.sraf_orders else '否'}_"
                            f"PVBand={'是' if self.pvband_cost else '否'}_"
                            f"固定宽度={'是' if self.fix_sraf else '否'}_"
                            f"{self.curve_type}_"
                            f"{self.file_name}"
                        )
        else:
            self.filename = (
                            f"骨架线参数化_"
                            f"宽度={self.initial_width}_"
                            f"分阶={'是' if self.sraf_orders else '否'}_"
                            f"PVBand={'是' if self.pvband_cost else '否'}_"
                            f"固定宽度={'是' if self.fix_sraf else '否'}_"
                            f"{self.curve_type}_"
                            f"{self.file_name}"
                        )
        # self._get_dir(self.up_filename, self.second_filename,self.curve_type,self.filename)
        # file_path = f'./{self.up_filename}/{self.second_filename}/{self.curve_type}/{self.filename}'
        # 换到路径为中期检查
        third_filename = '中期检查'
        self._get_dir(self.up_filename, self.second_filename,third_filename,self.filename)
        return str(Path(self.up_filename) / self.second_filename / third_filename / self.filename)
    
    

    def _load_srafsandinitial_mask(self):
        '''
        用于加载LSM优化后的版图，并将主图形和sraf图形分开
        切趾和无切趾只是文件命名可以忽略
        '''
        # ls_mask = np.loadtxt(f'./outputs/opc/CTM和levelset图像//Ls_mask/ls_image{self.pattern_name}.txt') #0.2tr
        if self.lsmispe_epe_pvband:
        # ls_mask = np.loadtxt(f'./outputs/opc/CTM和levelset图像/Ls_mask/ls_image{self.pattern_name}切趾.txt') #0.2tr
            ls_mask = np.loadtxt(case_1_result_path('Ls_mask', f'ls_image{self.pattern_name}pe_epe_pvband无切趾.txt')) #0.2tr
            # ls_mask = np.loadtxt(f'./outputs/opc/CTM和levelset图像/Ls_mask/ls_image{self.pattern_name}_cpupe_epe_pvband无切趾.txt')
        else:
            ls_mask = np.loadtxt(case_1_result_path('Ls_mask', f'ls_image{self.pattern_name}切趾.txt')) #0.2tr
        _,sraf = extract_sraf(ls_mask, self.target_mask)
          
        return ls_mask, sraf
    

    
   
    
    
    
    
        
