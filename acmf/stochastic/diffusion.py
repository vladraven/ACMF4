import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState


def compute_diffusion_sigma(
    state: StateVector,
    forcing: ForcingState,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет матрицу диффузии sigma(X) и ее частную производную d_sigma/d_SID.
    """
    sigma = np.zeros(params.N_sub, dtype=np.float64)
    d_sigma = np.zeros(params.N_sub, dtype=np.float64)

    denom = params.SID_buf * params.SID_max
    if denom <= 0.0:
        return sigma, d_sigma

    for k in range(params.N_sub):
        sid_k = state.sid[k]
        quad = (sid_k + params.SID_buf) * (params.SID_max - sid_k)
        if quad > 0.0:
            sigma[k] = params.sigma_0[k] * forcing.V * quad / denom
            d_sigma[k] = params.sigma_0[k] * forcing.V * (params.SID_max - params.SID_buf - 2.0 * sid_k) / denom
        else:
            sigma[k] = 0.0
            d_sigma[k] = 0.0

    return sigma, d_sigma