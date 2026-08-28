import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.bifurcation import ContinuationEngine
from acmf.analysis.jacobian import compute_dde_jacobians
from acmf.validation.result import TestResult


def run_test_05(params: ModelParameters) -> TestResult:
    """TEST 05 — Проверка delay-зависимости и обнаружение Hopf-перехода."""
    engine = ContinuationEngine()
    forcing = ForcingProfile().evaluate(0.0)

    def drift_factory(delay_val: float):
        def drift_det(x: np.ndarray) -> np.ndarray:
            st = StateVector(x)
            return compute_full_drift_vector(
                st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub), params
            )

        def drift_full(x_curr: np.ndarray, x_delayed: np.ndarray) -> np.ndarray:
            st = StateVector(x_curr)
            return compute_full_drift_vector(
                st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub), params
            )

        return drift_det, drift_full

    init_state = np.zeros(9 + params.N_sub + 1, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    delay_range = np.linspace(0.0, 5.0, 20)
    drift_det, drift_full = drift_factory(float(delay_range[0]))

    from acmf.analysis.equilibria import EquilibriumEngine

    equilibrium = EquilibriumEngine().find_equilibrium(
        drift_det,
        init_state,
    )

    if not equilibrium.is_valid:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan",
            status="FAILED",
            details={"reason": "equilibrium_not_found"},
        )

    a_0, a_1 = compute_dde_jacobians(
        drift_full,
        equilibrium.state,
    )

    delay_sensitivity = float(np.linalg.norm(a_1))

    if delay_sensitivity == 0.0:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan",
            status="FAILED",
            details={
                "reason": "delay_jacobian_is_zero",
                "delay_jacobian_norm": delay_sensitivity,
            },
        )

    bifurcations = engine.scan_parameter(
        drift_factory=drift_factory,
        param_values=delay_range,
        initial_state=init_state,
        delay=float(params.Delta_t),
    )

    hopf_points = [
        b for b in bifurcations
        if b.bifurcation_type == "Hopf"
    ]

    status = "PASSED" if hopf_points else "NOT_DETECTED"

    return TestResult(
        test_id="TEST_05",
        name="Hopf DDE Bifurcation Scan",
        status=status,
        details={
            "bifurcations_detected": len(bifurcations),
            "hopf_count": len(hopf_points),
            "delay_jacobian_norm": delay_sensitivity,
        },
    )
