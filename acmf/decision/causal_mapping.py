from dataclasses import dataclass
import numpy as np
from acmf.model.parameters import ModelParameters


@dataclass(frozen=True)
class PolicyIntervention:
    """Вектор регуляторных управляющих воздействий u."""
    u_reform: float = 0.0      # Аддитивный импульс реформ -> lambda_ref_0
    u_capacity: float = 0.0    # Модификатор производственных емкостей -> Capacity
    u_mitigation: float = 0.0  # Снижение уязвимости институтов -> omega_V

    def validate(self) -> None:
        if self.u_reform < 0.0:
            raise ValueError("u_reform должно быть >= 0")
        if self.u_capacity < 0.0:
            raise ValueError("u_capacity должно быть >= 0")
        if not (0.0 <= self.u_mitigation <= 1.0):
            raise ValueError("u_mitigation должно быть в [0, 1]")


def apply_causal_policy_mapping(
    params: ModelParameters,
    intervention: PolicyIntervention,
) -> ModelParameters:
    """
    Каузальное отображение параметров без прямого изменения SID:
    u_reform     -> lambda_ref_0
    u_capacity   -> Capacity_k
    u_mitigation -> omega_V
    """
    intervention.validate()

    new_lambda_ref = params.lambda_ref_0 + intervention.u_reform
    new_capacity = np.clip(params.Capacity * (1.0 + intervention.u_capacity), 0.0, 1.0)
    new_omega_v = max(0.0, params.omega_V * (1.0 - intervention.u_mitigation))

    return ModelParameters(
        N_sub=params.N_sub,
        F_max=params.F_max,
        SID_buf=params.SID_buf,
        SID_max=params.SID_max,
        kappa_s=params.kappa_s,
        alpha_pos=params.alpha_pos,
        beta_neg=params.beta_neg,
        gamma_inst=params.gamma_inst,
        mu_inst=params.mu_inst,
        alpha_F=params.alpha_F,
        beta_F=params.beta_F,
        alpha_Ch=params.alpha_Ch,
        beta_Ch=params.beta_Ch,
        mu_Ch=params.mu_Ch,
        alpha_Prod=params.alpha_Prod,
        beta_Prod=params.beta_Prod,
        alpha_M=params.alpha_M,
        mu_M=params.mu_M,
        gamma_scar=params.gamma_scar,
        mu_scar=params.mu_scar,
        Threshold_scar=params.Threshold_scar,
        gamma_R=params.gamma_R,
        theta_A=params.theta_A,
        theta_P=params.theta_P,
        theta_I=params.theta_I,
        w=params.w,
        p=params.p,
        eta=params.eta,
        Capacity=new_capacity,
        rho=params.rho,
        sigma_0=params.sigma_0,
        alpha_mask=params.alpha_mask,
        lambda_burst=params.lambda_burst,
        alpha_burst=params.alpha_burst,
        ED_crit=params.ED_crit,
        ED_scale=params.ED_scale,
        ED_impact=params.ED_impact,
        RefThresh=params.RefThresh,
        lambda_ref_0=new_lambda_ref,
        omega_fatigue=params.omega_fatigue,
        tau_ref=params.tau_ref,
        Delta_t=params.Delta_t,
        Delta_ref=params.Delta_ref,
        kappa_spill=params.kappa_spill,
        SID_contagion=params.SID_contagion,
        beta_H=params.beta_H,
        Gamma=params.Gamma,
        mu_rec=params.mu_rec,
        omega_V=new_omega_v,
        omega_SID=params.omega_SID,
    )