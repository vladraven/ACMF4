from typing import List
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class SystemDimensionsConfig(BaseModel):
    N_sub: int = Field(default=3, frozen=True, description="Фиксированное число подсистем")
    F_max: float = Field(..., gt=0.0)
    SID_buf: float = Field(..., gt=0.0)
    SID_max: float = Field(..., gt=0.0)
    kappa_s: float = Field(..., gt=0.0)


class DecayAndRatesConfig(BaseModel):
    alpha_pos: float = Field(..., ge=0.0)
    beta_neg: float = Field(..., ge=0.0)
    gamma_inst: float = Field(..., ge=0.0)
    mu_inst: float = Field(..., ge=0.0)
    alpha_F: float = Field(..., ge=0.0)
    beta_F: float = Field(..., ge=0.0)
    alpha_Ch: float = Field(..., ge=0.0)
    beta_Ch: float = Field(..., ge=0.0)
    mu_Ch: float = Field(..., ge=0.0)
    alpha_Prod: float = Field(..., ge=0.0)
    beta_Prod: float = Field(..., ge=0.0)
    alpha_M: float = Field(..., ge=0.0)
    mu_M: float = Field(..., ge=0.0)
    gamma_scar: float = Field(..., ge=0.0)
    mu_scar: float = Field(..., ge=0.0)
    Threshold_scar: float = Field(...)
    gamma_R: float = Field(..., ge=0.0, le=1.0)
    theta_A: float = Field(..., ge=0.0)
    theta_P: float = Field(..., ge=0.0)
    theta_I: float = Field(..., ge=0.0)


class MetabolismConfig(BaseModel):
    w: List[List[float]] = Field(..., description="Матрица весов износа 3x3")
    p: List[List[float]] = Field(..., description="Матрица генерации порядка 3x2")
    eta: float = Field(..., ge=0.0)
    Capacity: List[float] = Field(..., min_length=3, max_length=3)
    rho: List[float] = Field(..., min_length=3, max_length=3)
    sigma_0: List[float] = Field(..., min_length=3, max_length=3)

    @field_validator("Capacity", "rho", "sigma_0")
    @classmethod
    def check_non_negative_list(cls, v: List[float]) -> List[float]:
        if any(x < 0.0 for x in v):
            raise ValueError("Элементы списка должны быть >= 0")
        return v


class EpistemicAndReformConfig(BaseModel):
    alpha_mask: List[float] = Field(..., min_length=3, max_length=3)
    lambda_burst: float = Field(..., ge=0.0)
    alpha_burst: float = Field(..., ge=0.0)
    ED_crit: float = Field(..., ge=0.0)
    ED_scale: float = Field(..., gt=0.0)
    ED_impact: float = Field(..., ge=0.0)
    RefThresh: float = Field(...)
    lambda_ref_0: float = Field(..., ge=0.0)
    omega_fatigue: float = Field(..., ge=0.0, le=1.0)
    tau_ref: float = Field(..., ge=0.0)
    Delta_t: float = Field(..., ge=0.0)
    Delta_ref: float = Field(..., ge=0.0)


class ContagionAndHawkesConfig(BaseModel):
    kappa_spill: float = Field(..., ge=0.0)
    SID_contagion: float = Field(...)
    beta_H: float = Field(..., gt=0.0)
    Gamma: List[List[float]] = Field(...)
    mu_rec: float = Field(..., ge=0.0)
    omega_V: float = Field(..., ge=0.0)
    omega_SID: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_hawkes_spectral_radius(self) -> "ContagionAndHawkesConfig":
        gamma_mat = np.array(self.Gamma, dtype=np.float64)
        if gamma_mat.shape != (3, 3):
            raise ValueError("Матрица Gamma должна быть размером 3x3")
        if (gamma_mat < 0.0).any():
            raise ValueError("Элементы матрицы Gamma должны быть >= 0")
        
        k_h = gamma_mat / self.beta_H
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(k_h))))
        if spectral_radius >= 1.0:
            raise ValueError(
                f"Нарушено условие субкритичности Хоукса: rho(Gamma/beta_H) = {spectral_radius:.4f} >= 1.0"
            )
        return self


class ModelParametersSchema(BaseModel):
    """Корневая валидированная схема конфигурации ACMF 4.9.3.1."""
    dimensions: SystemDimensionsConfig
    dynamics: DecayAndRatesConfig
    metabolism: MetabolismConfig
    epistemic: EpistemicAndReformConfig
    contagion: ContagionAndHawkesConfig