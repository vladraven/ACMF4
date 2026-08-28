import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.smooth import s_plus


def compute_normalized_network_weights(raw_adj: np.ndarray) -> np.ndarray:
    """
    Нормирует направленную матрицу смежности сети J_ij по строкам без учета диагонали.
    Если узел изолирован, строка заполняется нулями.
    """
    n = raw_adj.shape[0]
    j_norm = np.zeros_like(raw_adj, dtype=np.float64)
    for i in range(n):
        row_mask = np.ones(n, dtype=bool)
        row_mask[i] = False
        row_sum = np.sum(raw_adj[i, row_mask])
        if row_sum > 0.0:
            j_norm[i, row_mask] = raw_adj[i, row_mask] / row_sum
    return j_norm


def compute_spatial_spillover(
    node_idx: int,
    all_sids: np.ndarray,
    j_matrix: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    Вычисляет пространственный переток дефицита:
    Spillover_i^k = kappa_spill * (SID_max - SID_i^k)/SID_max * sum_j [ J_ij * S+(SID_j^k - SID_contagion; kappa_s) ]
    all_sids имеет форму (N_nodes, 3)
    """
    sid_i = all_sids[node_idx]
    n_nodes = all_sids.shape[0]
    spillover = np.zeros(3, dtype=np.float64)

    if params.SID_max <= 0.0 or params.kappa_spill == 0.0 or n_nodes <= 1:
        return spillover

    boundary_factor = np.maximum(0.0, (params.SID_max - sid_i) / params.SID_max)

    for k in range(3):
        contagion_sum = 0.0
        for j in range(n_nodes):
            if i_idx := (j != node_idx):
                weight = j_matrix[node_idx, j]
                if weight > 0.0:
                    excess = float(s_plus(all_sids[j, k] - params.SID_contagion, params.kappa_s))
                    contagion_sum += weight * excess
        spillover[k] = params.kappa_spill * boundary_factor[k] * contagion_sum

    return spillover