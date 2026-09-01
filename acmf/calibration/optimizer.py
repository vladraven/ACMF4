from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import differential_evolution
from acmf.calibration.dataset import CalibrationDataset
from acmf.calibration.loss import CalibrationLoss
from acmf.model.parameters import ModelParameters


@dataclass(frozen=True)
class ParameterBounds:
    """Границы поиска для калибруемых параметров."""
    alpha_pos: tuple[float, float] = (0.2, 3.0)
    beta_neg: tuple[float, float] = (0.2, 3.0)
    gamma_inst: tuple[float, float] = (0.05, 1.0)
    mu_inst: tuple[float, float] = (0.01, 0.5)
    alpha_Prod: tuple[float, float] = (0.2, 2.5)
    beta_Prod: tuple[float, float] = (0.1, 1.5)


@dataclass
class CalibrationResult:
    """Итоговые метрики и оптимизированные параметры."""
    optimal_parameters: ModelParameters
    initial_loss: float
    final_loss: float
    r_squared: float
    success: bool
    iterations: int
    # ИСПРАВЛЕНО: раньше исключения при симуляции молча гасились в
    # CalibrationLoss без единого следа. Теперь статистика исключений
    # пробрасывается в результат калибровки, чтобы run_calibration.py
    # мог честно отчитаться, если бóльшая часть популяции DE не смогла
    # даже проинтегрироваться (см. exception_count/exception_reasons).
    exception_count: int
    exception_reasons: dict[str, int]


class EmpiricalOptimizer:
    """Движок глобальной оптимизации параметров под эмпирический ряд."""

    def __init__(
        self,
        dataset: CalibrationDataset,
        base_params: ModelParameters,
        bounds: ParameterBounds | None = None,
    ) -> None:
        self.dataset = dataset
        self.base_params = base_params
        self.bounds = bounds or ParameterBounds()
        self.loss_engine = CalibrationLoss(dataset, base_params)

    def _pack_vector_to_params(self, vec: np.ndarray) -> ModelParameters:
        """Переносит вектор оптимизации в неизменяемый ModelParameters."""
        return ModelParameters(
            N_sub=self.base_params.N_sub,
            F_max=self.base_params.F_max,
            SID_buf=self.base_params.SID_buf,
            SID_max=self.base_params.SID_max,
            kappa_s=self.base_params.kappa_s,
            alpha_pos=float(vec[0]),
            beta_neg=float(vec[1]),
            gamma_inst=float(vec[2]),
            mu_inst=float(vec[3]),
            alpha_F=self.base_params.alpha_F,
            beta_F=self.base_params.beta_F,
            alpha_Ch=self.base_params.alpha_Ch,
            beta_Ch=self.base_params.beta_Ch,
            mu_Ch=self.base_params.mu_Ch,
            alpha_Prod=float(vec[4]),
            beta_Prod=float(vec[5]),
            alpha_M=self.base_params.alpha_M,
            mu_M=self.base_params.mu_M,
            gamma_scar=self.base_params.gamma_scar,
            mu_scar=self.base_params.mu_scar,
            Threshold_scar=self.base_params.Threshold_scar,
            gamma_R=self.base_params.gamma_R,
            theta_A=self.base_params.theta_A,
            theta_P=self.base_params.theta_P,
            theta_I=self.base_params.theta_I,
            w=self.base_params.w,
            p=self.base_params.p,
            eta=self.base_params.eta,
            Capacity=self.base_params.Capacity,
            rho=self.base_params.rho,
            sigma_0=self.base_params.sigma_0,
            alpha_mask=self.base_params.alpha_mask,
            lambda_burst=self.base_params.lambda_burst,
            alpha_burst=self.base_params.alpha_burst,
            ED_crit=self.base_params.ED_crit,
            ED_scale=self.base_params.ED_scale,
            ED_impact=self.base_params.ED_impact,
            RefThresh=self.base_params.RefThresh,
            lambda_ref_0=self.base_params.lambda_ref_0,
            omega_fatigue=self.base_params.omega_fatigue,
            tau_ref=self.base_params.tau_ref,
            Delta_t=self.base_params.Delta_t,
            Delta_ref=self.base_params.Delta_ref,
            kappa_spill=self.base_params.kappa_spill,
            SID_contagion=self.base_params.SID_contagion,
            beta_H=self.base_params.beta_H,
            Gamma=self.base_params.Gamma,
            mu_rec=self.base_params.mu_rec,
            omega_V=self.base_params.omega_V,
            omega_SID=self.base_params.omega_SID,
        )

    def calibrate(self, maxiter: int = 15, popsize: int = 8) -> CalibrationResult:
        """Запускает дифференциальную эволюцию для нахождения глобального минимума ошибки."""
        param_bounds = [
            self.bounds.alpha_pos,
            self.bounds.beta_neg,
            self.bounds.gamma_inst,
            self.bounds.mu_inst,
            self.bounds.alpha_Prod,
            self.bounds.beta_Prod,
        ]

        def objective(vec: np.ndarray) -> float:
            candidate = self._pack_vector_to_params(vec)
            return self.loss_engine.evaluate(candidate)

        init_loss = self.loss_engine.evaluate(self.base_params)

        res = differential_evolution(
            func=objective,
            bounds=param_bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            polish=True,
        )

        opt_params = self._pack_vector_to_params(res.x)
        final_loss = float(res.fun)

        # Расчет R^2
        variance_obs = float(np.var(self.dataset.inst_obs))
        r2 = max(0.0, 1.0 - (final_loss / (variance_obs + 1e-8)))

        return CalibrationResult(
            optimal_parameters=opt_params,
            initial_loss=init_loss,
            final_loss=final_loss,
            r_squared=r2,
            success=bool(res.success),
            iterations=int(res.nit),
            exception_count=self.loss_engine.exception_count,
            exception_reasons=dict(self.loss_engine.exception_reasons),
        )