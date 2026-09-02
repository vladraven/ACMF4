from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
from acmf.solver.base import SolverDomain, DiagnosticStep
from acmf.solver.reflection import SkorokhodReflector
from acmf.solver.history import HistoryBuffer
from acmf.solver.delay import CausalDelayLookup
from acmf.solver.euler_maruyama import EulerMaruyamaStep
from acmf.solver.milstein import MilsteinStep
from acmf.model.forcing import ForcingProfile


@dataclass
class TrajectoryResult:
    """Результаты интегрирования полной траектории."""
    times: np.ndarray
    states: np.ndarray
    drifts: np.ndarray
    diagnostics: list[DiagnosticStep]


class ACMFEngine:
    """Оркестратор численного моделирования DDE-системы со скачками и отражением."""

    def __init__(
        self,
        domain: SolverDomain,
        scheme: Literal["euler_maruyama", "milstein"] = "euler_maruyama",
        state_dim: int = 13,
        sid_noise_dim: int = 3,
        buffer_capacity: int = 2000,
    ) -> None:
        self.reflector = SkorokhodReflector(domain)
        self.history = HistoryBuffer(capacity=buffer_capacity, state_dim=state_dim)
        self.step_scheme = MilsteinStep() if scheme == "milstein" else EulerMaruyamaStep()
        self.state_dim = state_dim
        self.sid_noise_dim = sid_noise_dim

    def simulate(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        drift_fn: Callable[[np.ndarray, float, float, float, float], np.ndarray],
        diffusion_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        jump_generator_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
        forcing_profile: ForcingProfile | None = None,
        delay_t: float = 0.0,
        delay_ref: float = 0.0,
        random_seed: int | None = None,
    ) -> TrajectoryResult:
        """Запускает цикл численной симуляции траектории."""
        rng = np.random.default_rng(random_seed)
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt)) + 1
        times = np.linspace(t_start, t_end, n_steps)

        states_history = np.zeros((n_steps, self.state_dim), dtype=np.float64)
        drifts_history = np.zeros((n_steps, self.state_dim), dtype=np.float64)
        diagnostics: list[DiagnosticStep] = []

        current_state, init_diag = self.reflector.reflect_state(initial_state)
        states_history[0] = current_state

        # Начальное forcing и dA/dt
        init_forcing = forcing_profile.evaluate(t_start) if forcing_profile is not None else None
        init_d_a_dt = float(init_forcing.dA_dt) if init_forcing is not None else 0.0

        initial_drift = drift_fn(current_state, 0.0, 0.0, 0.0, 0.0)
        drifts_history[0] = initial_drift
        self.history.append(t_start, current_state, initial_drift, d_a_dt=init_d_a_dt)

        for step_idx in range(n_steps - 1):
            t_curr = times[step_idx]

            # Текущее forcing записывается в history
            if forcing_profile is not None:
                forcing = forcing_profile.evaluate(t_curr)
                current_d_a_dt = float(forcing.dA_dt)
            else:
                current_d_a_dt = 0.0

            # === КАУЗАЛЬНЫЙ DELAY LOOKUP: читаем ЗАПАЗДЫВАЮЩИЕ производные из history ===
            if delay_t > 0.0 and self.history.size > 0:
                d_a_dt_delayed = CausalDelayLookup.get_delayed_forcing_derivative(
                    self.history, t_curr, delay_t
                )
                d_prod_dt = CausalDelayLookup.get_delayed_derivative(
                    self.history, t_curr, delay_t, component_idx=5
                )
                d_inst_dt = CausalDelayLookup.get_delayed_derivative(
                    self.history, t_curr, delay_t, component_idx=3
                )
            else:
                d_a_dt_delayed = current_d_a_dt
                d_prod_dt = 0.0
                d_inst_dt = 0.0

            if delay_ref > 0.0 and self.history.size > 0:
                d_agg_obs_dt = (
                    CausalDelayLookup.get_delayed_derivative(
                        self.history, t_curr, delay_ref, component_idx=0
                    )
                    + CausalDelayLookup.get_delayed_derivative(
                        self.history, t_curr, delay_ref, component_idx=1
                    )
                    + CausalDelayLookup.get_delayed_derivative(
                        self.history, t_curr, delay_ref, component_idx=2
                    )
                ) / 3.0
            else:
                d_agg_obs_dt = 0.0

            drift = drift_fn(current_state, d_a_dt_delayed, d_prod_dt, d_inst_dt, d_agg_obs_dt)
            sigma, d_sigma = diffusion_fn(current_state)

            jump = jump_generator_fn(current_state, t_curr) if jump_generator_fn else np.zeros(self.sid_noise_dim)

            dw_norm = rng.standard_normal(self.sid_noise_dim)

            raw_next_state = self.step_scheme.step(
                current_state=current_state,
                drift=drift,
                diffusion_sigma=sigma,
                diffusion_derivative=d_sigma,
                random_normal=dw_norm,
                jump_vector=jump,
                dt=dt,
            )

            next_state, diag = self.reflector.reflect_state(raw_next_state)

            states_history[step_idx + 1] = next_state
            drifts_history[step_idx] = drift
            diagnostics.append(diag)
            self.history.append(times[step_idx + 1], next_state, drift, d_a_dt=current_d_a_dt)

            current_state = next_state

        drifts_history[-1] = drift_fn(current_state, 0.0, 0.0, 0.0, 0.0)

        return TrajectoryResult(
            times=times,
            states=states_history,
            drifts=drifts_history,
            diagnostics=diagnostics,
        )