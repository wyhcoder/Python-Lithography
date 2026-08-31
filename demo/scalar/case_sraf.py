import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from op_model.demo_sraf import SRAF_Optimizer
from utils_model.project_paths import SCALAR_CONFIG_PATH

def main():
    
    #步骤1：从配置文件加载参数
    
    params = SimulationParameters.from_yaml(str(SCALAR_CONFIG_PATH))
    
    #步骤2：创建并初始化仿真器实例
    litho_simulator = LithographySimulator(params)
    
    litho_simulator.prepare_for_optimization(method="SOCS")
    
    sraf_optimizer = SRAF_Optimizer(simulator=litho_simulator)
    sraf_optimizer.run()
    
    
if __name__ == '__main__':
    main()
