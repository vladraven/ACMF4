import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState
from acmf.model.smooth import s_plus


def clip_jump_to_domain(
    current_sid: np.ndarray,
    raw_jump: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """Зажимает скачок в пределах домена."""
    clipped = np.zeros_like(raw_jump, dtype=np.float64)
    for k in range(params.N_sub):
        j_val = raw_jump[k]
        sid_val = current_sid[k]
        if j_val > 0.0:
            clipped[k] = min(j_val, params.SID_max - sid_val)
        elif j_val < 0.0:
            clipped[k] = max(j_val, -params.SID_buf - sid_val)
        else:
            clipped[k] = 0.0
    return clipped


def compute_recovery_debt_cost(
    jump_vec: np.ndarray,
    state: StateVector,
    forcing: ForcingState,
    agg_sid_true: float,
    params: ModelParameters,
) -> float:
    """Вычисляет прирост мобилизационного долга K_rec."""
    norm_j = float(np.linalg.norm(jump_vec, ord=2))
    if norm_j == 0.0:
        return 0.0

    sid_impact = float(s_plus(agg_sid_true, params.kappa_s))
    factor = 1.0 + params.omega_V * forcing.V + params.omega_SID * sid_impact
    return float(norm_j * factor)


def compute_recovery_debt_drift(state: StateVector, params: ModelParameters) -> float:
    """Вычисляет непрерывный дрейф мобилизационного долга dRecDebt/dt = -mu_rec * RecDebt."""
    return float(-params.mu_rec * state.rec_debt)