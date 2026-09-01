import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState


def compute_r_eff(r_0: float, scar: float, gamma_r: float) -> float:
    """Вычисляет эффективную регенерацию R_eff = R_0 * (1 - gamma_R * Scar)."""
    return float(r_0 * (1.0 - gamma_r * scar))


def compute_true_wear(
    state: StateVector,
    forcing: ForcingState,
    tsm: float,
    params: ModelParameters,
) -> np.ndarray:
    """Вычисляет истинный износ подсистем W_true (N_sub-вектор)."""
    w = params.w
    w_true = np.zeros(params.N_sub, dtype=np.float64)

    w_true[0] = w[0, 0] * (1.0 - state.inst) + w[0, 1] * (1.0 - state.ch) + w[0, 2] * tsm
    if params.N_sub > 1:
        w_true[1] = w[1, 0] * (1.0 - state.prod) + w[1, 1] * (1.0 - state.inst)
    if params.N_sub > 2:
        f_wear = (params.F_max - state.f) / params.F_max if params.F_max > 0.0 else 0.0
        w_true[2] = w[2, 0] * f_wear + w[2, 1] * (1.0 - state.m)

    return np.maximum(0.0, w_true)


def compute_order_generation(
    state: StateVector,
    forcing: ForcingState,
    r_eff: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет генерацию порядка Q_k и эффективную емкость ECY_k."""
    p = params.p
    q = np.zeros(params.N_sub, dtype=np.float64)
    q[0] = p[0, 0] * state.inst * r_eff + p[0, 1] * state.ch
    if params.N_sub > 1:
        q[1] = p[1, 0] * state.prod + p[1, 1] * forcing.A
    if params.N_sub > 2:
        q[2] = p[2, 0] * state.m + p[2, 1] * forcing.G

    q = np.maximum(0.0, q)
    q_tilde = q / (1.0 + params.eta * state.rec_debt)
    ecy = params.Capacity * (1.0 - np.exp(-q_tilde))

    return q, ecy