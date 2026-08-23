import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  
from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v1, images_simulation_v2
from utils_model.plot_matrix import plot_matrix,save_mask_as_bmp
import os
from op_model.demo_pe_gradient import compute_mask_pe_gradient
from op_model.demo_pe_ctm import CTM_Optimizer
from utils_model.project_paths import OPC_OUTPUT_DIR, SCALAR_CONFIG_PATH


def main():
    """
    主执行函数
    """
    
    print("--- STAGE 1: INITIALIZING THE LITHOGRAPHY SYSTEM ---")
    
    # 步骤 1: 从配置文件加载参数（跨平台路径）
    params = SimulationParameters.from_yaml(str(SCALAR_CONFIG_PATH))

    
    # 步骤 2: 创建并初始化仿真器实例
    litho_simulator = LithographySimulator(params)

    # ---display Data ---
    
    litho_simulator.prepare_for_optimization(method='SOCS')
    plot_matrix(litho_simulator.source.source_map,'Source')
    plot_matrix(litho_simulator.mask.data,'Target Patterns')
    plot_matrix(litho_simulator.optics.pupil_function, 'Puiple')
    
    # --- end ---
    
    # ---display EPE points---
    
    import matplotlib.pyplot as plt

    midpoints = litho_simulator.mask.midpoints
    # print(midpoints)

    # plt.figure(figsize=(10, 10))

    # # 使用 origin='upper' 参数，这是最关键的修改
    # plt.imshow(litho_simulator.mask.data, cmap='gray', interpolation='nearest', 
    #         alpha=0.8, origin='upper') 

    # if midpoints:
    #     # midpoints 的坐标是 (row, col) 即 (y, x)，这样解包是正确的
    #     mid_y, mid_x = zip(*midpoints)
    #     plt.scatter(mid_x, mid_y, c='red', s=100, marker='x', label='Line Midpoints')

    # plt.title("Key Point Extraction Demo")
    # plt.legend()



    # plt.show()
    
    # --- end ---
    
    # ---display images---
    initial_mask = litho_simulator.get_current_mask_spatial()
    
    aerial_image, wafer_image, intermediates = images_simulation_v2(
        mask_spatial=initial_mask,
        resist_params=litho_simulator.params.resist,
        opt_cache=litho_simulator.opt_cache
    )
    
    plot_matrix(aerial_image)
    plot_matrix(wafer_image)
    
    Gm, pe_erro = compute_mask_pe_gradient(wafer_image, initial_mask, 
                                intermediates, 
                                litho_simulator.opt_cache, 
                                litho_simulator.params.resist)
    
    print(f"Gradient calculated with shape: {Gm.shape}")
    print(f"\nL2 Error: {pe_erro}")
    
    
    print("\n--- STAGE 2: CREATING THE OPTIMIZER ---")
    
    # STEP 1: CTM
    ctm_optimizer = CTM_Optimizer(simulator=litho_simulator, iterations=50, learning_rate=1)
    final_op_mask = ctm_optimizer.run()
    plot_matrix(final_op_mask,'final_op_mask')
    
    # STEP 2: Extract SRAFs
    from utils_model.demo_mask_binarization import binarize_mask

    _, _, srafs, final, zone = binarize_mask(
        gray_mask=final_op_mask,
        target_pattern=initial_mask,
        quantile_val=0.125,
        area_threshold=50, #50
        peak_min_intensity=0.1,
        min_spacing_pixels=10   # distance for Main pattern
    )
    base_dir = OPC_OUTPUT_DIR / "CTM和levelset图像"
    name =   f"CTM_image{params.mask.image_name}"
    img_ctm_dir = base_dir / "CTM_mask"
    img_ctm_dir.mkdir(parents=True, exist_ok=True)
    save_path_png = img_ctm_dir / f"{name}.png"
    plot_matrix(srafs,'srafs')
    plot_matrix(final,'final')
    plt.imsave(save_path_png, final, cmap="gray")
    plot_matrix(zone,'zone')
    
    # STEP 3: Level set Methods
    
    litho_simulator.update_mask(final)
    from op_model.demo_level_set import LevelSetOptimizer
    lsm_optimizer = LevelSetOptimizer(litho_simulator, dx=4, dy=4, cfl=0.5, b=0.0000001, ltr=0.0)
    best_mask, min_error = lsm_optimizer.run(50,5) # 50次迭代，逢5次修正RSD，（越频繁效果越好）
    plot_matrix(best_mask)
    
    txt_dir  = base_dir / "Ls_mask"
    img_dir  = base_dir / "Ls_mask图像"
    png_dir = img_dir / "png"
    bmp_dir = img_dir / "BMP"
    pe_dir = img_dir / "PE"
    txt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    bmp_dir.mkdir(parents=True, exist_ok=True)
    pe_dir.mkdir(parents=True, exist_ok=True)
    name = f"ls_image{params.mask.image_name}"

    save_path_txt = txt_dir / f"{name}.txt"
    save_path_png = png_dir / f"{name}.png"
    save_path_bmp = bmp_dir / f"{name}.bmp"
    save_path_pe = pe_dir / f"{name}.txt"
    # -------- PNG（仅用于可视化） --------
    plt.imsave(save_path_png, best_mask, cmap="gray")

    # # -------- BMP（工程 / 写版 / EDA 用） --------
    save_mask_as_bmp(best_mask, save_path_bmp)

    # -------- TXT（数值存档） --------
    np.savetxt(
        save_path_txt,
        (best_mask > 0).astype(np.uint8),  # 明确转整数
        fmt="%d",
        delimiter=" "
    )
    # -------- PE（物理存档） --------
    if os.path.isdir(save_path_pe):
        os.rmdir(save_path_pe)
    with open(save_path_pe, "w") as f:
        f.write(f"{name}\n")
        f.write(f"PE_error:{min_error:.6f}\n")
        
if __name__ == "__main__":
    main()
