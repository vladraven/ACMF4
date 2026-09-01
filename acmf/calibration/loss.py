import logging
from collections import Counter
from typing import Callable
import numpy as np
from acmf.calibration.dataset import CalibrationDataset
from acmf.model.parameters import ModelParameters
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.state import StateVector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine

logger = logging.getLogger("acmf.calibration.loss")


class CalibrationLoss:
    """Функция потерь для подгонки параметров под эмпирический датасет."""

    def __init__(
        self,
        dataset: CalibrationDataset,
        base_params: ModelParameters,
        forcing_profile: ForcingProfile | None = None,
    ) -> None:
        self.dataset = dataset
        self.base_params = base_params
        self.forcing = (forcing_profile or ForcingProfile()).evaluate(0.0)
        # ИСПРАВЛЕНО: раньше любое исключение при интегрировании молча
        # гасилось и подменялось фиксированным штрафом 1e6, без единой
        # записи о том, что вообще произошло. Если numerical blow-up
        # происходил для большей части популяции differential evolution,
        # ландшафт функции потерь становился плоским, DE "сходился" за
        # 1 итерацию — не найдя минимум, а исчерпав градиентный сигнал.
        # Теперь каждое исключение логируется с типом и сообщением, и
        # ведётся счётчик/последняя причина для отчёта по итогам калибровки.
        self.exception_count: int = 0
        self.exception_reasons: Counter[str] = Counter()
        self.last_exception_message: str | None = None

    def evaluate(self, candidate_params: ModelParameters) -> float:
        """Вычисляет взвешенную MSE между траекторией модели и наблюдаемыми рядами."""
        domain = SolverDomain(
            sid_buf=candidate_params.SID_buf,
            sid_max=candidate_params.SID_max,
            f_max=candidate_params.F_max,
        )
        engine = ACMFEngine(domain=domain, scheme="euler_maruyama")

        t_span = (float(self.dataset.times[0]), float(self.dataset.times[-1]))
        dt = float(self.dataset.times[1] - self.dataset.times[0])

        init_state = np.zeros(13, dtype=np.float64)
        init_state[0:3] = self.dataset.sid_obs[0]
        init_state[3] = self.dataset.inst_obs[0]
        init_state[5] = self.dataset.prod_obs[0]
        init_state[4] = 0.8  # Ch
        init_state[6] = 0.8  # M
        init_state[7] = 0.8 * candidate_params.F_max

        def drift_fn(x, d_a, d_p, d_i, d_agg):
            st = StateVector(x)
            return compute_full_drift_vector(
                st, self.forcing, d_a, d_p, d_i, d_agg, np.zeros(3), candidate_params
            )

        def diff_fn(x):
            return np.zeros(3), np.zeros(3)

        try:
            traj = engine.simulate(
                initial_state=init_state,
                t_span=t_span,
                dt=dt,
                drift_fn=drift_fn,
                diffusion_fn=diff_fn,
            )
        except Exception as exc:
            self.exception_count += 1
            reason = f"{type(exc).__name__}: {exc}"
            self.exception_reasons[type(exc).__name__] += 1
            self.last_exception_message = reason
            logger.warning(
                "CalibrationLoss: симуляция упала для кандидата параметров "
                "(alpha_pos=%.4f, beta_neg=%.4f, gamma_inst=%.4f, mu_inst=%.4f, "
                "alpha_Prod=%.4f, beta_Prod=%.4f): %s",
                candidate_params.alpha_pos, candidate_params.beta_neg,
                candidate_params.gamma_inst, candidate_params.mu_inst,
                candidate_params.alpha_Prod, candidate_params.beta_Prod,
                reason,
            )
            return 1e6  # Штраф за численную неустойчивость — теперь с логированием причины

        # Выравнивание размерностей при несовпадении сеток
        min_len = min(len(self.dataset.times), len(traj.times))
        sim_sid = traj.states[:min_len, 0:3]
        sim_inst = traj.states[:min_len, 3]
        sim_prod = traj.states[:min_len, 5]

        obs_sid = self.dataset.sid_obs[:min_len]
        obs_inst = self.dataset.inst_obs[:min_len]
        obs_prod = self.dataset.prod_obs[:min_len]

        # Расчет взвешенной ошибки
        err_sid = np.mean((sim_sid - obs_sid)**2, axis=0)  # (3,)
        err_inst = np.mean((sim_inst - obs_inst)**2)
        err_prod = np.mean((sim_prod - obs_prod)**2)

        total_err_vec = np.array([err_sid[0], err_sid[1], err_sid[2], err_inst, err_prod])
        weighted_loss = float(np.sum(self.dataset.weights * total_err_vec))

        # Регуляризация за нарушение границ
        if not StateVector(traj.states[-1]).is_in_domain(candidate_params):
            weighted_loss += 100.0

        return weighted_loss

        # Выравнивание размерностей при несовпадении сеток
        min_len = min(len(self.dataset.times), len(traj.times))
        sim_sid = traj.states[:min_len, 0:3]
        sim_inst = traj.states[:min_len, 3]
        sim_prod = traj.states[:min_len, 5]

        obs_sid = self.dataset.sid_obs[:min_len]
        obs_inst = self.dataset.inst_obs[:min_len]
        obs_prod = self.dataset.prod_obs[:min_len]

        # Расчет взвешенной ошибки
        err_sid = np.mean((sim_sid - obs_sid)**2, axis=0)  # (3,)
        err_inst = np.mean((sim_inst - obs_inst)**2)
        err_prod = np.mean((sim_prod - obs_prod)**2)

        total_err_vec = np.array([err_sid[0], err_sid[1], err_sid[2], err_inst, err_prod])
        weighted_loss = float(np.sum(self.dataset.weights * total_err_vec))

        # Регуляризация за нарушение границ
        if not StateVector(traj.states[-1]).is_in_domain(candidate_params):
            weighted_loss += 100.0

        return weighted_loss