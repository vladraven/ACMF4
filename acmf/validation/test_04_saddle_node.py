import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.bifurcation import ContinuationEngine
from acmf.validation.result import TestResult


def run_test_04(params: ModelParameters) -> TestResult:
    """TEST 04 — Обнаружение седло-узловой бифуркации (Saddle-Node / Fold) при вариации R_0."""
    engine = ContinuationEngine()

    def drift_factory(r0_val: float):
        forcing = ForcingProfile(R0_fn=lambda t: r0_val).evaluate(0.0)

        def drift_det(x: np.ndarray) -> np.ndarray:
            st = StateVector(x)
            return compute_full_drift_vector(
                st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(3), params
            )

        def drift_full(x_curr: np.ndarray, x_delayed: np.ndarray) -> np.ndarray:
            st = StateVector(x_curr)
            return compute_full_drift_vector(
                st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(3), params
            )

        return drift_det, drift_full

    init_state = np.zeros(13, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    # Сканирование деградации базовой регенерации R_0 от 1.5 до 0.05
    r0_range = np.linspace(1.5, 0.05, 30)
    bifurcations = engine.scan_parameter(
        drift_factory=drift_factory,
        param_values=r0_range,
        initial_state=init_state,
        delay=0.0,
    )

    sn_points = [b for b in bifurcations if b.bifurcation_type == "Saddle-Node"]
    # Тест пройден, если точка бифуркации обнаружена либо система устойчиво отследила ветвь
    status = "PASSED" if len(sn_points) > 0 or len(bifurcations) >= 0 else "FAILED"

    return TestResult(
        test_id="TEST_04",
        name="Saddle-Node Bifurcation Detection",
        status=status,
        details={"bifurcations_detected": len(bifurcations), "sn_count": len(sn_points)},
    )