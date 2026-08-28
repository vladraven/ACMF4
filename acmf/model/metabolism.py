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
    """
    Вычисляет истинный износ подсистем W_true (3-вектор):
    W^1 = w_11*(1 - Inst) + w_12*(1 - Ch) + w_13*TSM
    W^2 = w_21*(1 - Prod) + w_22*(1 - Inst)
    W^3 = w_31*(F_max - F)/F_max + w_32*(1 - M)
    """
    w = params.w
    w_true = np.zeros(3, dtype=np.float64)

    w_true[0] = w[0, 0] * (1.0 - state.inst) + w[0, 1] * (1.0 - state.ch) + w[0, 2] * tsm
    w_true[1] = w[1, 0] * (1.0 - state.prod) + w[1, 1] * (1.0 - state.inst)
    f_wear = (params.F_max - state.f) / params.F_max if params.F_max > 0.0 else 0.0
    w_true[2] = w[2, 0] * f_wear + w[2, 1] * (1.0 - state.m)

    return np.maximum(0.0, w_true)


def compute_order_generation(
    state: StateVector,
    forcing: ForcingState,
    r_eff: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет генерацию порядка Q_k, Q_tilde_k и эффективную емкость ECY_k:
    Q_1 = p_11 * Inst * R_eff + p_12 * Ch
    Q_2 = p_21 * Prod + p_22 * A
    Q_3 = p_31 * M + p_32 * G
    Q_tilde_k = Q_k / (1 + eta * RecDebt)
    ECY^k = Capacity_k * [1 - exp(-Q_tilde_k)]
    """
    p = params.p
    q = np.zeros(3, dtype=np.float64)
    q[0] = p[0, 0] * state.inst * r_eff + p[0, 1] * state.ch
    q[1] = p[1, 0] * state.prod + p[1, 1] * forcing.A
    q[2] = p[2, 0] * state.m + p[2, 1] * forcing.G

    q = np.maximum(0.0, q)
    q_tilde = q / (1.0 + params.eta * state.rec_debt)
    ecy = params.Capacity * (1.0 - np.exp(-q_tilde))

    return q, ecy