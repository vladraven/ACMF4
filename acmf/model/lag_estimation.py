from __future__ import annotations

import numpy as np

from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.epistemic import compute_aggregated_sids


def estimate_lagged_derivatives(
    x_curr: np.ndarray,
    x_delayed: np.ndarray,
    delay: float,
    params: ModelParameters,
    d_a_dt: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Оценивает запаздывающие производные dInst/dt, dProd/dt, dAggSID_obs/dt
    по конечной разности между текущим и запаздывающим состоянием:
        d/dt X(t - Delta) ~= [X(t) - X(t - Delta)] / Delta

    Используется в местах, где нет полного history buffer (равновесный /
    bifurcation-scan контекст), но честная delay-зависимость всё равно
    требуется для непустого A_1 = dF/dX(t - Delta_t) в DDE-Якобиане.

    d_a_dt передаётся отдельно, так как A(t) — экзогенное воздействие,
    а не компонента state vector: при постоянном forcing-профиле
    (ForcingProfile с константными функциями) его производная равна 0
    по определению, а не по недосмотру.
    """
    if delay <= 0.0:
        return d_a_dt, 0.0, 0.0, 0.0

    st_curr = StateVector(x_curr)
    st_delayed = StateVector(x_delayed)

    d_inst_dt = (st_curr.inst - st_delayed.inst) / delay
    d_prod_dt = (st_curr.prod - st_delayed.prod) / delay

    agg_true_curr, agg_obs_curr = compute_aggregated_sids(st_curr, params)
    agg_true_delayed, agg_obs_delayed = compute_aggregated_sids(st_delayed, params)
    d_agg_obs_dt = (agg_obs_curr - agg_obs_delayed) / delay

    return d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt
