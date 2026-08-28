from dataclasses import dataclass
import numpy as np
from config.schema import ModelParametersSchema


@dataclass(frozen=True)
class ModelParameters:
    """Неизменяемый контейнер параметров для вычислительного ядра ACMF."""
    N_sub: int
    F_max: float
    SID_buf: float
    SID_max: float
    kappa_s: float
    alpha_pos: float
    beta_neg: float
    gamma_inst: float
    mu_inst: float
    alpha_F: float
    beta_F: float
    alpha_Ch: float
    beta_Ch: float
    mu_Ch: float
    alpha_Prod: float
    beta_Prod: float
    alpha_M: float
    mu_M: float
    gamma_scar: float
    mu_scar: float
    Threshold_scar: float
    gamma_R: float
    theta_A: float
    theta_P: float
    theta_I: float
    w: np.ndarray
    p: np.ndarray
    eta: float
    Capacity: np.ndarray
    rho: np.ndarray
    sigma_0: np.ndarray
    alpha_mask: np.ndarray
    lambda_burst: float
    alpha_burst: float
    ED_crit: float
    ED_scale: float
    ED_impact: float
    RefThresh: float
    lambda_ref_0: float
    omega_fatigue: float
    tau_ref: float
    Delta_t: float
    Delta_ref: float
    kappa_spill: float
    SID_contagion: float
    beta_H: float
    Gamma: np.ndarray
    mu_rec: float
    omega_V: float
    omega_SID: float

    @classmethod
    def from_schema(cls, schema: ModelParametersSchema) -> "ModelParameters":
        return cls(
            N_sub=schema.dimensions.N_sub,
            F_max=schema.dimensions.F_max,
            SID_buf=schema.dimensions.SID_buf,
            SID_max=schema.dimensions.SID_max,
            kappa_s=schema.dimensions.kappa_s,
            alpha_pos=schema.dynamics.alpha_pos,
            beta_neg=schema.dynamics.beta_neg,
            gamma_inst=schema.dynamics.gamma_inst,
            mu_inst=schema.dynamics.mu_inst,
            alpha_F=schema.dynamics.alpha_F,
            beta_F=schema.dynamics.beta_F,
            alpha_Ch=schema.dynamics.alpha_Ch,
            beta_Ch=schema.dynamics.beta_Ch,
            mu_Ch=schema.dynamics.mu_Ch,
            alpha_Prod=schema.dynamics.alpha_Prod,
            beta_Prod=schema.dynamics.beta_Prod,
            alpha_M=schema.dynamics.alpha_M,
            mu_M=schema.dynamics.mu_M,
            gamma_scar=schema.dynamics.gamma_scar,
            mu_scar=schema.dynamics.mu_scar,
            Threshold_scar=schema.dynamics.Threshold_scar,
            gamma_R=schema.dynamics.gamma_R,
            theta_A=schema.dynamics.theta_A,
            theta_P=schema.dynamics.theta_P,
            theta_I=schema.dynamics.theta_I,
            w=np.array(schema.metabolism.w, dtype=np.float64),
            p=np.array(schema.metabolism.p, dtype=np.float64),
            eta=schema.metabolism.eta,
            Capacity=np.array(schema.metabolism.Capacity, dtype=np.float64),
            rho=np.array(schema.metabolism.rho, dtype=np.float64),
            sigma_0=np.array(schema.metabolism.sigma_0, dtype=np.float64),
            alpha_mask=np.array(schema.epistemic.alpha_mask, dtype=np.float64),
            lambda_burst=schema.epistemic.lambda_burst,
            alpha_burst=schema.epistemic.alpha_burst,
            ED_crit=schema.epistemic.ED_crit,
            ED_scale=schema.epistemic.ED_scale,
            ED_impact=schema.epistemic.ED_impact,
            RefThresh=schema.epistemic.RefThresh,
            lambda_ref_0=schema.epistemic.lambda_ref_0,
            omega_fatigue=schema.epistemic.omega_fatigue,
            tau_ref=schema.epistemic.tau_ref,
            Delta_t=schema.epistemic.Delta_t,
            Delta_ref=schema.epistemic.Delta_ref,
            kappa_spill=schema.contagion.kappa_spill,
            SID_contagion=schema.contagion.SID_contagion,
            beta_H=schema.contagion.beta_H,
            Gamma=np.array(schema.contagion.Gamma, dtype=np.float64),
            mu_rec=schema.contagion.mu_rec,
            omega_V=schema.contagion.omega_V,
            omega_SID=schema.contagion.omega_SID,
        )