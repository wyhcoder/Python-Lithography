import numpy as np
from pathlib import Path
from skimage import draw, measure
from utils_model.project_paths import OPC_OUTPUT_DIR
from utils_model.select_eps import DemoSelectEps_new
from scipy.ndimage import  binary_dilation
from .load_mask_bmp import load_mask_image
from matplotlib import pyplot as plt
from skimage.morphology import  disk
import cv2
class MEEF:
    """
    根据MEEF参数生成输入的参数化版图
    """
    def __init__(self, 
                 target_mask: np.ndarray,
                 curve_type: str,
                 simplt_type: str,
                 val_simply: float,
                 pattern_name:str,
                 file_name:str,
                 ifOPC: bool,
                 SRAF: bool,
                 move_strategy: str = "bisector"):
        
        self.target_mask = target_mask.copy() # 目标版图
        self.move_strategy = move_strategy    # 控制点移动策略: bisector / xy
        # np.savetxt(F"ls_image{pattern_name}.txt", self.target_mask, fmt='%d', delimiter=' ')
        # print("LSMmask生成完成")
        # plt.imshow(self.target_mask, cmap='gray')
        # plt.show()
        self.curve_type = curve_type     # 参数化曲线的类型
        self.simplt_type = simplt_type   # 抽稀方法
        self.val_simply = val_simply     # 抽稀参数
        self.pattern_name = pattern_name # 版图名称
        self.file_name = file_name       # 自定义的文件名称
        self.mid_weight = 4              # 关键点权重
        self.orther_weight = 1           # 非关键点权重
        self.interval_line = 5          # EP点取点每个点之间的间隔 
        self.interval_corner = 2         # EP点部分距离角点的距离
        self.delta = 0.15                # 差分时控制点的移动量，原本0.15     
        self.ifOPC = ifOPC               # 主图形控制点是否直接在目标版图上取
        self.SRAF = SRAF                 # 是否是对初始化宽度的SRAF进行的主图形控制点位置的优化，以降低EPE为目的
        # 类方法，用于打EP点
        # self.select_epsclass = DemoSelectEps(self.target_mask, self.mid_weight, self.orther_weight,self.pattern_name)
        self.select_epsclass = DemoSelectEps_new(self.target_mask, self.mid_weight, self.orther_weight,self.pattern_name)
        self.filepath = self._get_file() # 获取保存文件路径
        
        # 获取EP点，wepe_caculated用于计算矩阵，对原始meef矩阵进行二次处理的weight_meef
        self.eps, self.wepe_caculated, self.weight_meef = self._select_ep_points() 
        np.savetxt(f"{self.filepath}/select_ep_points.txt",np.array(self.eps),fmt='%d')
        self.num_eps =  len(self.eps) # EP点的数量
        self.num_weps = np.sum(np.array(self.wepe_caculated) == 1) # wEP点的数量
        
        # self.initial_mask是LSM生成的初始掩膜
        if  not self.SRAF:
            # 依据抽稀方式，加载抽稀后的LSMmask和SRAF
            self.lsm_mask, _ = self._load_srafsandinitial_mask()
        else:
            # 加载初始化宽度的LSMmask (用正斜杠, Windows/macOS 兼容)
            self.lsm_mask, _ = self._load_srafsandinitial_mask()
            self.initial_mask = np.loadtxt(
                OPC_OUTPUT_DIR / 'sraf_width_opt' / '宽度初始化后的LSMmask'
                / f'{self.file_name}分阶后有主图形初始化宽度.txt'
            )
        
            
        self._load_simply_sraf()  # 加载抽稀处理后的SRAF
        self.num_cps = sum(len(c) for c in self.cps) # 控制点的数量
        
    
    
    # 获取文件路径的函数
    def _get_file(self)-> str:
        """
        创建保存结果的文件夹
        """
        self.up_filename = str(OPC_OUTPUT_DIR)
        if self.ifOPC:
            self.second_filename = 'OPC'
        else:
            self.second_filename = self.simplt_type
        self.third_filename = f'{self.simplt_type}_data'
        self.filename = f'{self.val_simply}_{self.curve_type}_{self.file_name}'
        # if self.ifOPC:
        #     intverval = 10  # OPC的间隔固定为10
        #     self.filename = f'{self.simplt_type}{self.val_simply}_主图形间隔{intverval}{self.curve_type}_{self.file_name}'
        self._get_dir(self.up_filename, self.second_filename,self.curve_type,self.filename)
        return str(Path(self.up_filename) / self.second_filename / self.curve_type / self.filename)
    
    # 创建保存文件夹的函数 
    def _get_dir(self,up_filename: str,sc_filename: str,curve_type: str,filename: str)-> None:
        base_dir = Path(up_filename) / sc_filename/curve_type/filename
        base_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择EP点的函数，根据图形类型选择不同的EP点
    def _select_ep_points(self)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pattern_name == '中心对称通孔' or self.pattern_name == '对角通孔':
            # 对于通孔类的版图，每边中点为关键点
            eps,wepe_caculated, weight_meef = self.select_epsclass._select_eps_ofvia(r = 2)
        else:
            # 对于没有明显对称性的图形，其关键EP点为每边EP点的75%
            eps,wepe_caculated, weight_meef = self.select_epsclass._select_eps_ofothers(self.interval_line, self.interval_corner)                
            self.select_epsclass.debug_plot_eps(eps,wepe_caculated,weight_meef,self.filepath)
        return eps,wepe_caculated, weight_meef
    
    # 加载LSM生成的初始掩膜和SRAF的函数，返回LSM掩模和SRAF掩模
    def _load_srafsandinitial_mask(self)-> tuple[np.ndarray, np.ndarray]:
        ls_mask = np.loadtxt(
            OPC_OUTPUT_DIR / 'CTM和levelset图像' / 'Ls_mask'
            / f'ls_image{self.pattern_name}.txt'
        ) #0.2tr
        sraf = self.extract_sraf(ls_mask, self.target_mask)    
        return ls_mask, sraf    

    # 加载抽稀处理后的SRAF的函数，根据抽稀方法和参数加载不同的SRAF掩模，并根据EP点重新排序控制点的顺序    
    def _load_simply_sraf(self):
        
        if self.simplt_type == 'DPS':
            _ , self.sraf_mask = self.extract_sraf(np.loadtxt(Path(self.up_filename) / self.second_filename / self.third_filename / 'LSmask' / f'{self.pattern_name}{self.curve_type}{self.val_simply}.txt'), self.target_mask)    
            plt.imshow(self.sraf_mask)
            plt.show()
            self.initial_cps = np.load(Path(self.up_filename) / self.second_filename / self.third_filename / 'CPS' / f'{self.pattern_name}{self.val_simply}.npy', allow_pickle=True)
            print('加载的DPS的srafs')
            
        if self.simplt_type == 'US':
            _ , self.sraf_mask = self.extract_sraf(np.loadtxt(Path(self.up_filename) / self.second_filename / self.third_filename / 'LSmask' / f'{self.pattern_name}{self.curve_type}{self.val_simply}.txt'), self.target_mask)
            print('加载的US的srafs')
            # if self.pattern_name == '中心对称通孔' or self.pattern_name == '对角通孔':
            #     self.initial_cps = self.select_epsclass.extract_mask_control_points(5,symmetry="left-right")
            # else:
            #     self.initial_cps = self.select_epsclass.extract_mask_control_points(4)
            self.initial_cps = np.load(Path(self.up_filename) / self.second_filename / self.third_filename / 'CPS' / f'{self.pattern_name}{self.val_simply}.npy', allow_pickle=True)
            # self.initial_cps = self.select_epsclass.extract_mask_control_points(8)
        if self.simplt_type == 'CS':
            _ , self.sraf_mask = self.extract_sraf(np.loadtxt(Path(self.up_filename) / self.second_filename / self.third_filename / 'LSmask' / f'{self.pattern_name}{self.curve_type}{self.val_simply}.txt'), self.target_mask)
            self.initial_cps = np.load(Path(self.up_filename) / self.second_filename / self.third_filename / 'CPS' / f'{self.pattern_name}{self.val_simply}.npy',allow_pickle=True)
            print('加载的CS的srafs')
        if self.ifOPC:
            if self.pattern_name == '中心对称通孔':
                self.initial_cps = self.select_epsclass.extract_mask_control_points(5,symmetry="left-right")   #有对称
            elif self.pattern_name == '对角通孔':
                self.initial_cps = self.select_epsclass.extract_mask_control_points(5,symmetry="diagonal")   #有对称
            else:
                self.initial_cps = self.select_epsclass.extract_mask_control_points(5,symmetry=None)   #无对称
        if self.SRAF:
            _, self.sraf_mask = self.extract_sraf(
                np.loadtxt(
                    OPC_OUTPUT_DIR / 'sraf_width_opt' / '宽度初始化后的LSMmask'
                    / f'{self.file_name}分阶后有主图形初始化宽度.txt'
                ),
                self.target_mask,
            )
            print('加载OPC的分阶后有主图形初始化宽度srafs')
        self.cps = self.reorder_list1(self.initial_cps, self.eps)
     
    # 根据EP点重新排序控制点的顺序，确保与EP点的距离较近的控制点在前面
    def reorder_list1(self,cps_list: list, eps_list: list)-> np.ndarray:
        """
        根据list1子列表与list2的平均距离调整顺序
        """
        if len(cps_list) != 2:
            return cps_list
        
        # 计算两个子列表与list2的平均距离
        distance_a = self.calculate_sum_distance(cps_list[0], eps_list[0])
        distance_b = self.calculate_sum_distance(cps_list[1], eps_list[0])
        
        # 比较距离，决定顺序
        return cps_list if distance_a <= distance_b else [cps_list[1], cps_list[0]] 
    
    # 计算一个子列表中所有点与另一个列表中所有点的距离之和
    def calculate_sum_distance(self, cps_list, eps_list):
      
        # 统一转换为numpy数组（处理可能的numpy array或普通列表）
        cps_arr = np.array([point.tolist() if isinstance(point, np.ndarray) else point for point in cps_list])
        eps_arr = np.array(eps_list)
        
        # 计算所有点对的欧氏距离矩阵（形状：[len(cps_arr), len(eps_arr)]）
        distances = np.linalg.norm(cps_arr[:, np.newaxis] - eps_arr, axis=2)
        
        # 返回所有距离的和（而非平均值）
        return np.sum(distances)  
      
    # 分离主图形和SRAF的函数        
    
    def extract_sraf(
            self,
            full_mask: np.ndarray,
            origin_mask: np.ndarray,
            dilate_radius: int = 3,
            fg_thresh: float = 1e-6,
            overlap_ratio: float = 0.05,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        从 full_mask 中分离主图形和 SRAF，保留原灰度值。

        改进点（相对旧版"仅按 origin 膨胀区域分"）：
        - 对 full_mask 做前景连通域标记 (8 邻接)；
        - 每个连通域按"与 origin_mask 的真重叠"判断归属：
            * 重叠 >= max(1px, overlap_ratio * 域面积) 视为主图形；
            * 否则视为 SRAF。
        - 旧版的"膨胀 origin 当主图形区域"只用作回退/兜底，
            不再决定边界归属，避免主图形偏移/形变后边缘被错判为 SRAF。

        参数:
            full_mask:     包含主图形和 SRAF 的完整掩模 (可为灰度)。
            origin_mask:   只含主图形的参考掩模 (>0 视为前景)。
            dilate_radius: 旧接口兼容参数；当 origin 与 full_mask 主图形
                        几乎完全无重叠时，会用此半径膨胀 origin 来兜底。
            fg_thresh:     前景判定阈值（灰度像素 >fg_thresh 视为前景）。
            overlap_ratio: 一个连通域被判为主图形的最小重叠占比。

        返回:
            main_mask, sraf_mask : 与 full_mask 同 dtype，同灰度值。
        """
        full_mask = np.asarray(full_mask)
        full_f = full_mask.astype(np.float32, copy=False)
        fg = (full_f > fg_thresh).astype(np.uint8)

        origin_bin = (np.asarray(origin_mask) > 0).astype(np.uint8)

        main_mask = np.zeros_like(full_mask)
        sraf_mask = np.zeros_like(full_mask)

        if fg.sum() == 0:
            return main_mask, sraf_mask

        # 1) 连通域标记（8 邻接）+ 按"真重叠面积"决定归属
        num, labels = cv2.connectedComponents(fg, connectivity=8)
        matched_any = False
        for lbl in range(1, num):
            comp = (labels == lbl)
            area = int(comp.sum())
            if area == 0:
                continue
            overlap = int(np.logical_and(comp, origin_bin).sum())
            # 主图形判据：与 origin 真实有像素重叠，且占比 >= overlap_ratio
            is_main = (overlap > 0) and (overlap >= max(1, int(overlap_ratio * area)))
            if is_main:
                main_mask[comp] = full_mask[comp]
                matched_any = True
            else:
                sraf_mask[comp] = full_mask[comp]

        # 2) 兜底：若连通域分析没能匹配到任何主图形（origin 与 full_mask 完全
        #    不重叠的极端情况），退化为旧版逻辑"膨胀 origin 当主图形区域"。
        if not matched_any:
            dilated_origin = binary_dilation(origin_bin, structure=disk(dilate_radius))
            main_mask[:] = 0
            sraf_mask[:] = 0
            main_mask[dilated_origin] = full_mask[dilated_origin]
            sraf_mask[~dilated_origin] = full_mask[~dilated_origin]

        return main_mask, sraf_mask
        
    
