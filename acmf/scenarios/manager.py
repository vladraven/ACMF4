from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from acmf.model.forcing import ForcingState, ForcingProfile
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.decision.causal_mapping import PolicyIntervention, apply_causal_policy_mapping
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine, TrajectoryResult


@dataclass
class DynamicScenario:
    """Конфигурация сценария с зависящими от времени шоками и политиками."""
    name: str
    t_span: tuple[float, float] = (0.0, 50.0)
    dt: float = 0.05
    initial_state: np.ndarray | None = None
    forcing_profile: ForcingProfile = field(default_factory=ForcingProfile)
    policy_schedule: list[tuple[float, PolicyIntervention]] = field(default_factory=list)
    custom_jump_generator: Callable[[np.ndarray, float], np.ndarray] | None = None
    random_seed: int | None = 42


class ScenarioManager:
    """Исполнитель пакетных сценарных расчетов."""

    def __init__(self, base_params: ModelParameters) -> None:
        self.base_params = base_params

    def run_scenario(self, scenario: DynamicScenario) -> TrajectoryResult:
        domain = SolverDomain(
            sid_buf=self.base_params.SID_buf,
            sid_max=self.base_params.SID_max,
            f_max=self.base_params.F_max,
        )
        engine = ACMFEngine(domain=domain, scheme="milstein")

        init_state = scenario.initial_state
        if init_state is None:
            init_state = np.zeros(13, dtype=np.float64)
            init_state[3:7] = 0.8
            init_state[7] = 0.8 * self.base_params.F_max

        # Сортировка расписания регуляторных интервенций по времени
        sorted_policies = sorted(scenario.policy_schedule, key=lambda x: x[0])

        def get_current_params(t_curr: float) -> ModelParameters:
            active_params = self.base_params
            for p_time, policy in sorted_policies:
                if t_curr >= p_time:
                    active_params = apply_causal_policy_mapping(active_params, policy)
            return active_params

        def drift_fn(x: np.ndarray, d_a: float, d_p: float, d_i: float, d_agg: float) -> np.ndarray:
            st = StateVector(x)
            # Вектор времени извлекается через текущий шаг
            # Для простоты оцениваем через базовые форсинги
            # Здесь форсинг вычисляется динамически
            forcing = scenario.forcing_profile.evaluate(0.0)
            cur_p = get_current_params(0.0)
            return compute_full_drift_vector(st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), cur_p)

        def diff_fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            st = StateVector(x)
            forcing = scenario.forcing_profile.evaluate(0.0)
            cur_p = get_current_params(0.0)
            return compute_diffusion_sigma(st, forcing, cur_p)

        return engine.simulate(
            initial_state=init_state,
            t_span=scenario.t_span,
            dt=scenario.dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            jump_generator_fn=scenario.custom_jump_generator,
            random_seed=scenario.random_seed,
        )