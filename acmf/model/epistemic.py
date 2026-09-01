import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState


def compute_visibility_gap(forcing: ForcingState, params: ModelParameters) -> np.ndarray:
    """Вычисляет разрыв видимости VisibilityGap^k = alpha_mask^k * (1 - V) * (1 - G)."""
    return params.alpha_mask * (1.0 - forcing.V) * (1.0 - forcing.G)


def compute_observed_wear(w_true: np.ndarray, visibility_gap: np.ndarray) -> np.ndarray:
    """Вычисляет наблюдаемый износ W_obs^k = W_true^k * (1 - VisibilityGap^k)."""
    return w_true * (1.0 - visibility_gap)


def compute_burst_rate(ed_k: float, params: ModelParameters) -> float:
    """Вычисляет скорость взрывного раскрытия долга B_burst(ED^k)."""
    arg = -params.alpha_burst * (ed_k - params.ED_crit)
    return float(params.lambda_burst / (1.0 + np.exp(arg)))


def compute_epistemic_debt_drift(
    state: StateVector,
    w_true: np.ndarray,
    w_obs: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """Вычисляет dED^k/dt = (W_true^k - W_obs^k) - B_burst(ED^k) * ED^k."""
    ed = state.ed
    d_ed = np.zeros(params.N_sub, dtype=np.float64)
    for k in range(params.N_sub):
        b_burst = compute_burst_rate(float(ed[k]), params)
        d_ed[k] = (w_true[k] - w_obs[k]) - b_burst * ed[k]
    return d_ed


def compute_aggregated_sids(state: StateVector, params: ModelParameters) -> tuple[float, float]:
    """
    Вычисляет истинный и наблюдаемый агрегированный дефицит:
    AggSID_true = sum(SID^k) / N_sub
    AggSID_obs = AggSID_true - ED_impact * sum(ED_norm^k) / N_sub
    """
    agg_true = float(np.sum(state.sid) / params.N_sub)
    ed_norm = state.ed / (state.ed + params.ED_scale)
    agg_obs = float(agg_true - (params.ED_impact * np.sum(ed_norm) / params.N_sub))
    return agg_true, agg_obs