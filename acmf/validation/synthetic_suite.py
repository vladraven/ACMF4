import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.forcing import ForcingProfile
from acmf.decision.causal_mapping import PolicyIntervention
from acmf.scenarios.manager import DynamicScenario, ScenarioManager
from acmf.solver.engine import TrajectoryResult
from acmf.analysis.jacobian import compute_finite_difference_jacobian
from acmf.analysis.spectrum import analyze_instantaneous_spectrum
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.ews.projection import ModalProjection
from acmf.ews.variance import compute_rolling_variance
from acmf.ews.autocorrelation import compute_rolling_ar1
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.state import StateVector


class SyntheticTestSuite:
    """Генератор синтетических стресс-тестов для верификации и отображения на дашборде."""

    def __init__(self, params: ModelParameters) -> None:
        self.params = params
        self.manager = ScenarioManager(params)

    def _compute_ews_payload(self, traj: TrajectoryResult) -> dict[str, list[float]]:
        """Вспомогательный расчет модальной проекции Z, скользящей дисперсии и автокорреляции."""
        eq_engine = EquilibriumEngine()
        forcing = ForcingProfile().evaluate(0.0)

        def drift_fn(x):
            st = StateVector(x)
            return compute_full_drift_vector(st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(3), self.params)

        guess = np.zeros(13, dtype=np.float64)
        guess[3:7] = 0.8
        guess[7] = 0.8 * self.params.F_max
        eq = eq_engine.find_equilibrium(drift_fn, guess)

        if not eq.is_valid:
            return {"z_series": [], "variance": [], "ar1": []}

        jac = compute_finite_difference_jacobian(drift_fn, eq.state)
        spec = analyze_instantaneous_spectrum(jac)

        z = ModalProjection.compute_trajectory_projection(traj.states, eq.state, spec.left_critical_vector)
        var_z = compute_rolling_variance(z, window_size=30)
        ar1_z = compute_rolling_ar1(z, window_size=30)

        return {
            "z_series": [float(val) for val in z],
            "variance": [float(val) for val in var_z],
            "ar1": [float(val) for val in ar1_z],
        }

    def run_flash_crash(self) -> dict:
        """Сценарий 1: Flash Crash Shock."""
        def jump_fn(x, t):
            if 3.0 <= t <= 3.1:
                return np.array([1.5, 0.2, 0.0])
            return np.zeros(3)

        scenario = DynamicScenario(
            name="Flash Crash Shock",
            t_span=(0.0, 25.0),
            dt=0.05,
            custom_jump_generator=jump_fn,
            random_seed=101,
        )
        traj = self.manager.run_scenario(scenario)
        ews = self._compute_ews_payload(traj)

        recovered = bool(traj.states[-1, 0] < 0.2 and traj.states[-1, 3] > 0.6)

        return {
            "id": "SYNTH_01",
            "name": scenario.name,
            "passed": recovered,
            "trajectory": traj,
            "ews": ews,
            "summary": "Резкий импульсный шок институционального дефицита с полной релаксацией в базовый аттрактор.",
        }

    def run_cascading_failure(self) -> dict:
        """Сценарий 2: Cascading Failure."""
        def jump_fn(x, t):
            if 2.0 <= t <= 2.2:
                return np.array([1.8, 1.6, 1.2])
            return np.zeros(3)

        scenario = DynamicScenario(
            name="Cascading Multilateral Failure",
            t_span=(0.0, 30.0),
            dt=0.05,
            custom_jump_generator=jump_fn,
            random_seed=202,
        )
        traj = self.manager.run_scenario(scenario)
        ews = self._compute_ews_payload(traj)

        scar_accumulated = bool(traj.states[-1, 8] > 0.3)

        return {
            "id": "SYNTH_02",
            "name": scenario.name,
            "passed": scar_accumulated,
            "trajectory": traj,
            "ews": ews,
            "summary": "Каскадный коллапс институтов и производства с фиксацией структурного гистерезисного шрама Scar.",
        }

    def run_ews_pre_collapse(self) -> dict:
        """Сценарий 3: EWS Pre-Collapse Degradation."""
        scenario = DynamicScenario(
            name="EWS Degradation Monitor",
            t_span=(0.0, 35.0),
            dt=0.05,
            random_seed=303,
        )
        traj = self.manager.run_scenario(scenario)
        ews = self._compute_ews_payload(traj)

        ews_active = len(ews["variance"]) > 0 and not np.isnan(ews["variance"][-1])

        return {
            "id": "SYNTH_03",
            "name": scenario.name,
            "passed": ews_active,
            "trajectory": traj,
            "ews": ews,
            "summary": "Непрерывный мониторинг модальной проекции Z(t), скользящей дисперсии и автокорреляции AR(1).",
        }

    def run_policy_rescue(self) -> dict:
        """Сценарий 4: Policy Rescue Intervention."""
        def jump_fn(x, t):
            if 2.0 <= t <= 2.2:
                return np.array([1.6, 1.4, 0.5])
            return np.zeros(3)

        policies = [
            (5.0, PolicyIntervention(u_reform=1.0, u_capacity=0.6, u_mitigation=0.5))
        ]

        scenario = DynamicScenario(
            name="Policy Rescue Intervention",
            t_span=(0.0, 30.0),
            dt=0.05,
            policy_schedule=policies,
            custom_jump_generator=jump_fn,
            random_seed=404,
        )
        traj = self.manager.run_scenario(scenario)
        ews = self._compute_ews_payload(traj)

        rescued = bool(traj.states[-1, 3] > 0.75)

        return {
            "id": "SYNTH_04",
            "name": scenario.name,
            "passed": rescued,
            "trajectory": traj,
            "ews": ews,
            "summary": "Купирование глубокого системного кризиса через включение регуляторных реформ u(t) на шаге t=5.",
        }

    def run_all(self) -> list[dict]:
        return [
            self.run_flash_crash(),
            self.run_cascading_failure(),
            self.run_ews_pre_collapse(),
            self.run_policy_rescue(),
        ]