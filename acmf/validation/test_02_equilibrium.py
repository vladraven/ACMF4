import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.validation.result import TestResult


def run_test_02(params: ModelParameters) -> TestResult:
    """TEST 02 — Поиск равновесий и мультистабильности."""
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()

    def drift_fn(x):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub), params
        )

    state_dim = 2 * params.N_sub + 7

    def domain_sampler() -> np.ndarray:
        rng = np.random.default_rng()
        guess = np.zeros(state_dim, dtype=np.float64)
        guess[0:params.N_sub] = rng.uniform(-params.SID_buf, params.SID_max, params.N_sub)
        guess[params.N_sub : params.N_sub + 4] = rng.uniform(0.0, 1.0, 4)
        guess[params.N_sub + 4] = rng.uniform(0.0, params.F_max)
        guess[params.N_sub + 5] = rng.uniform(0.0, 1.0)
        return guess

    guess1 = np.zeros(state_dim, dtype=np.float64)
    guess1[params.N_sub : params.N_sub + 4] = 0.8
    guess1[params.N_sub + 4] = 0.8 * params.F_max

    guess2 = np.zeros(state_dim, dtype=np.float64)
    guess2[0:params.N_sub] = 1.5
    guess2[params.N_sub : params.N_sub + 4] = 0.3
    guess2[params.N_sub + 4] = 0.3 * params.F_max

    eqs = eq_engine.scan_multistability(
        drift_fn,
        initial_guesses=[guess1, guess2],
        domain_sampler=domain_sampler,
        n_samples=50,
    )

    residuals = [float(eq.residual_norm) for eq in eqs]

    status = "PASSED" if len(eqs) > 0 else "FAILED"

    return TestResult(
        test_id="TEST_02",
        name="Equilibria and Multistability Scan",
        status=status,
        details={
            "num_equilibria_found": len(eqs),
            "residuals": residuals,
        },
    )