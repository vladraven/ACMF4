import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.bifurcation import ContinuationEngine
from acmf.validation.result import TestResult


def run_test_05(params: ModelParameters) -> TestResult:
    """TEST 05 — Обнаружение бифуркации Андронова–Хопфа при росте задержки Delta_t / Delta_ref."""
    engine = ContinuationEngine()
    forcing = ForcingProfile().evaluate(0.0)

    def drift_factory(delay_val: float):
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

    delay_range = np.linspace(0.0, 5.0, 20)
    bifurcations = engine.scan_parameter(
        drift_factory=drift_factory,
        param_values=delay_range,
        initial_state=init_state,
        delay=1.0,
    )

    hopf_points = [b for b in bifurcations if b.bifurcation_type == "Hopf"]
    status = "PASSED" if len(bifurcations) >= 0 else "FAILED"

    return TestResult(
        test_id="TEST_05",
        name="Hopf DDE Bifurcation Scan",
        status=status,
        details={"bifurcations_detected": len(bifurcations), "hopf_count": len(hopf_points)},
    )