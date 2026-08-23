import numpy as np

def caculate_epe(aI: np.ndarray, tr:float, eps: np.ndarray, nums_eps:int, weight:np.ndarray,add_weight:bool=False):
    dx = np.gradient(aI,1,axis=1)
    dy = np.gradient(aI,1,axis=0)
    G_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    epe_matrix = (((aI - tr) / G_magnitude)) ** 2
    epe_vector = np.array(
        [epe_matrix[row, col] for row, col in eps],
        dtype=np.float64
        )
    if add_weight:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape[0] != epe_vector.shape[0]:
                raise ValueError(
                f"weight 长度 ({weight.shape[0]}) 与 EPE 点数 ({epe_vector.shape[0]}) 不一致"
            )
        epe_vector *= weight
    epe_vector = np.round(epe_vector, 6)
    
    return np.round(np.sum(epe_vector), 6), epe_vector

def caculate_wepe(wepe_calculate: np.ndarray, epe_vector: np.ndarray, num_weps: int):
    wepe = np.asarray(wepe_calculate, dtype=float)
    epe  = np.asarray(epe_vector, dtype=float)
    weighted_epe = wepe * epe
    # weighted_epe /= num_weps
    return np.round(np.sum(weighted_epe),6)
    