import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.jacobian import compute_finite_difference_jacobian
from acmf.analysis.spectrum import analyze_instantaneous_spectrum
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.validation.result import TestResult


def run_test_03(params: ModelParameters) -> TestResult:
    """TEST 03 — Спектральная устойчивость Якобиана и проекция левых/правых мод."""
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()

    def drift_fn(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(3), params
        )

    guess = np.zeros(13, dtype=np.float64)
    guess[3:7] = 0.8
    guess[7] = 0.8 * params.F_max

    eq = eq_engine.find_equilibrium(drift_fn, guess)
    if not eq.is_valid:
        return TestResult(
            test_id="TEST_03",
            name="Instantaneous Jacobian Stability",
            status="FAILED",
            error_message="Не удалось найти базовую точку равновесия",
        )

    jac = compute_finite_difference_jacobian(drift_fn, eq.state)
    spec = analyze_instantaneous_spectrum(jac)

    # Проверка биортогональной нормировки w^H * v = 1
    normalization_error = float(abs(np.vdot(spec.left_critical_vector, spec.right_critical_vector) - 1.0))
    is_valid_normalization = normalization_error < 1e-4

    status = "PASSED" if spec.is_stable and is_valid_normalization else "FAILED"

    return TestResult(
        test_id="TEST_03",
        name="Instantaneous Jacobian Stability",
        status=status,
        details={
            "critical_eigenvalue": str(spec.critical_eigenvalue),
            "is_stable": spec.is_stable,
            "biorthogonal_norm_err": normalization_error,
        },
    )