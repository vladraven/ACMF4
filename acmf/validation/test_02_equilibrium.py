import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.equilibria import EquilibriumEngine, EquilibriumConfig
from acmf.validation.result import TestResult

DEFAULT_RESIDUAL_TOL: float = 1e-5


def run_test_02(params: ModelParameters) -> TestResult:
    """TEST 02 — Поиск и устойчивость стационарных состояний F(X*) = 0."""
    forcing = ForcingProfile().evaluate(0.0)
    config = EquilibriumConfig(residual_tol=DEFAULT_RESIDUAL_TOL)
    engine = EquilibriumEngine(config=config)

    def drift_deterministic(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(3), params
        )

    guess_healthy = np.zeros(13, dtype=np.float64)
    guess_healthy[3:7] = 0.8
    guess_healthy[7] = 0.8 * params.F_max

    guess_degraded = np.zeros(13, dtype=np.float64)
    guess_degraded[0:3] = 1.5
    guess_degraded[3:7] = 0.2
    guess_degraded[7] = 0.2 * params.F_max
    guess_degraded[8] = 0.6

    equilibria = engine.scan_multistability(drift_deterministic, [guess_healthy, guess_degraded])

    all_in_domain = all(StateVector(eq.state).is_in_domain(params) for eq in equilibria)
    status = "PASSED" if len(equilibria) >= 1 and all_in_domain else "FAILED"

    details = {
        "num_equilibria_found": len(equilibria),
        "residuals": [float(eq.residual_norm) for eq in equilibria],
    }

    return TestResult(
        test_id="TEST_02",
        name="Equilibria and Multistability Scan",
        status=status,
        details=details,
    )