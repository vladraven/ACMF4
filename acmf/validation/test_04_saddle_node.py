import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.lag_estimation import estimate_lagged_derivatives
from acmf.analysis.bifurcation import ContinuationEngine
from acmf.validation.result import TestResult


def run_test_04(params: ModelParameters) -> TestResult:
    """TEST 04 — Обнаружение седло-узловой бифуркации при вариации R_0."""
    engine = ContinuationEngine()

    def drift_factory(r0_val: float):
        forcing = ForcingProfile(R0_fn=lambda t: r0_val).evaluate(0.0)

        def drift_det(x: np.ndarray) -> np.ndarray:
            st = StateVector(x)
            return compute_full_drift_vector(
                st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub), params
            )

        def drift_full(x_curr: np.ndarray, x_delayed: np.ndarray) -> np.ndarray:
            # ИСПРАВЛЕНО: x_delayed раньше игнорировался (см. TEST_05).
            # Здесь r0_val — параметр continuation-скана (не задержка),
            # поэтому лаг для оценки производных берём из params.Delta_t.
            st = StateVector(x_curr)
            d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt = estimate_lagged_derivatives(
                x_curr, x_delayed, float(params.Delta_t), params
            )
            return compute_full_drift_vector(
                st, forcing, d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt,
                np.zeros(params.N_sub), params
            )

        return drift_det, drift_full

    init_state = np.zeros(9 + params.N_sub + 1, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    r0_range = np.linspace(1.5, 0.05, 30)
    bifurcations = engine.scan_parameter(
        drift_factory=drift_factory,
        param_values=r0_range,
        initial_state=init_state,
        delay=0.0,
    )

    sn_points = [
        b for b in bifurcations
        if b.bifurcation_type == "Saddle-Node"
    ]

    if not bifurcations:
        status = "NOT_DETECTED"
    else:
        status = "PASSED" if sn_points else "FAILED"

    return TestResult(
        test_id="TEST_04",
        name="Saddle-Node Bifurcation Detection",
        status=status,
        details={
            "bifurcations_detected": len(bifurcations),
            "sn_count": len(sn_points),
            "scan_parameter_min": float(r0_range.min()),
            "scan_parameter_max": float(r0_range.max()),
        },
    )
