import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState
from acmf.model.smooth import s_plus, s_minus
from acmf.model.metabolism import compute_r_eff, compute_true_wear, compute_order_generation
from acmf.model.epistemic import (
    compute_visibility_gap,
    compute_observed_wear,
    compute_epistemic_debt_drift,
    compute_aggregated_sids,
)
from acmf.model.jumps import compute_recovery_debt_drift


def compute_tsm(
    delayed_d_a_dt: float,
    delayed_d_prod_dt: float,
    delayed_d_inst_dt: float,
    params: ModelParameters,
) -> float:
    """
    Вычисляет Time-Scale Mismatch с запаздыванием:
    TSM = 1 - exp(- theta_A*|dA/dt| - theta_P*|dProd/dt| - theta_I*|dInst/dt|)
    """
    exponent = (
        - params.theta_A * abs(delayed_d_a_dt)
        - params.theta_P * abs(delayed_d_prod_dt)
        - params.theta_I * abs(delayed_d_inst_dt)
    )
    return float(1.0 - np.exp(exponent))


def compute_reform_impulse(
    agg_sid_obs: float,
    delayed_d_agg_sid_obs_dt: float,
    scar: float,
    forcing: ForcingState,
    params: ModelParameters,
) -> float:
    """
    Вычисляет реформаторский импульс:
    Awareness = S+(AggSID_obs - RefThresh; kappa_s)
    ReformImpulse = lambda_ref_0 * (1 - omega_fatigue * Scar) * Awareness * G * exp(-tau_ref * S+(dAggSID_obs/dt; kappa_s))
    """
    awareness = float(s_plus(agg_sid_obs - params.RefThresh, params.kappa_s))
    fatigue = max(0.0, 1.0 - params.omega_fatigue * scar)
    d_agg_smoothed = float(s_plus(delayed_d_agg_sid_obs_dt, params.kappa_s))
    delay_suppression = float(np.exp(-params.tau_ref * d_agg_smoothed))

    return float(params.lambda_ref_0 * fatigue * awareness * forcing.G * delay_suppression)


def compute_sid_drift(
    state: StateVector,
    w_true: np.ndarray,
    ecy: np.ndarray,
    r_eff: float,
    spillover: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    Вычисляет дрейф компонент SID:
    Drift^k = S+(Delta^k) * (SID_max - SID^k)/SID_max + S-(Delta^k) * (SID_buf + SID^k)/SID_buf
              - rho_k * R_eff * Inst * S+(SID^k) + Spillover^k
    """
    delta = w_true - ecy
    drift = np.zeros(3, dtype=np.float64)

    for k in range(3):
        sid_k = state.sid[k]
        term_pos = float(s_plus(delta[k], params.kappa_s)) * (params.SID_max - sid_k) / params.SID_max
        term_neg = float(s_minus(delta[k], params.kappa_s)) * (params.SID_buf + sid_k) / params.SID_buf
        reg_term = params.rho[k] * r_eff * state.inst * float(s_plus(sid_k, params.kappa_s))
        drift[k] = term_pos + term_neg - reg_term + spillover[k]

    return drift


def compute_bounded_ode_drifts(
    state: StateVector,
    forcing: ForcingState,
    r_eff: float,
    reform_impulse: float,
    agg_sid_true: float,
    tsm: float,
    params: ModelParameters,
) -> tuple[float, float, float, float, float, float]:
    """Вычисляет d/dt для [Inst, Ch, Prod, M, F, Scar]."""
    # 1. dInst/dt
    inst_growth = params.alpha_pos * (r_eff * state.ch + params.gamma_inst * state.m * forcing.G) + reform_impulse
    inst_decay = params.mu_inst + params.beta_neg * float(s_plus(agg_sid_true, params.kappa_s))
    d_inst = inst_growth * (1.0 - state.inst) - inst_decay * state.inst

    # 2. dF/dt
    d_f = (
        params.alpha_F * state.m * forcing.G * (params.F_max - state.f)
        - params.beta_F * float(s_plus(state.sid[2], params.kappa_s)) * state.f
    )

    # 3. dCh/dt
    d_ch = (
        params.alpha_Ch * state.inst * state.prod * (1.0 - state.ch)
        - (params.mu_Ch + params.beta_Ch * tsm) * state.ch
    )

    # 4. dProd/dt
    d_prod = (
        params.alpha_Prod * forcing.A * state.ch * (1.0 - state.prod)
        - params.beta_Prod * float(s_plus(state.sid[1], params.kappa_s)) * state.prod
    )

    # 5. dM/dt
    d_m = params.alpha_M * state.prod * state.inst * (1.0 - state.m) - params.mu_M * state.m

    # 6. dScar/dt
    scar_growth = params.gamma_scar * float(s_plus(agg_sid_true - params.Threshold_scar, params.kappa_s))
    d_scar = scar_growth * (1.0 - state.scar) - params.mu_scar * state.scar

    return d_inst, d_ch, d_prod, d_m, d_f, d_scar


def compute_full_drift_vector(
    state: StateVector,
    forcing: ForcingState,
    delayed_d_a_dt: float,
    delayed_d_prod_dt: float,
    delayed_d_inst_dt: float,
    delayed_d_agg_sid_obs_dt: float,
    spillover: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    Вычисляет полный 13-мерный детерминированный вектор дрейфа F(X).
    Возвращает плоский ndarray формы (13,).
    """
    r_eff = compute_r_eff(forcing.R_0, state.scar, params.gamma_R)
    tsm = compute_tsm(delayed_d_a_dt, delayed_d_prod_dt, delayed_d_inst_dt, params)
    w_true = compute_true_wear(state, forcing, tsm, params)
    _, ecy = compute_order_generation(state, forcing, r_eff, params)

    vis_gap = compute_visibility_gap(forcing, params)
    w_obs = compute_observed_wear(w_true, vis_gap)
    agg_true, agg_obs = compute_aggregated_sids(state, params)

    reform_impulse = compute_reform_impulse(agg_obs, delayed_d_agg_sid_obs_dt, state.scar, forcing, params)

    # SID дрейфы
    sid_drift = compute_sid_drift(state, w_true, ecy, r_eff, spillover, params)

    # ОДУ дрейфы
    d_inst, d_ch, d_prod, d_m, d_f, d_scar = compute_bounded_ode_drifts(
        state, forcing, r_eff, reform_impulse, agg_true, tsm, params
    )

    # ED дрейфы
    d_ed = compute_epistemic_debt_drift(state, w_true, w_obs, params)

    # RecDebt дрейф
    d_rec_debt = compute_recovery_debt_drift(state, params)

    full_drift = np.zeros(13, dtype=np.float64)
    full_drift[0:3] = sid_drift
    full_drift[3] = d_inst
    full_drift[4] = d_ch
    full_drift[5] = d_prod
    full_drift[6] = d_m
    full_drift[7] = d_f
    full_drift[8] = d_scar
    full_drift[9:12] = d_ed
    full_drift[12] = d_rec_debt

    return full_drift