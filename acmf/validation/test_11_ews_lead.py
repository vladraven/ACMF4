import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.jacobian import compute_finite_difference_jacobian
from acmf.analysis.spectrum import analyze_instantaneous_spectrum
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.ews.projection import ModalProjection
from acmf.ews.variance import compute_rolling_variance
from acmf.ews.autocorrelation import compute_rolling_ar1
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_11(params: ModelParameters) -> TestResult:
    """TEST 11 — Опережающие сигналы EWS."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama", state_dim=2 * params.N_sub + 7)
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(st, forcing, d_a, d_p, d_i, d_agg, np.zeros(params.N_sub), params)

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    state_dim = 2 * params.N_sub + 7
    guess = np.zeros(state_dim, dtype=np.float64)
    guess[params.N_sub : params.N_sub + 4] = 0.8
    guess[params.N_sub + 4] = 0.8 * params.F_max

    eq = eq_engine.find_equilibrium(lambda x: drift_fn(x, 0.0, 0.0, 0.0, 0.0), guess)

    jac = compute_finite_difference_jacobian(lambda x: drift_fn(x, 0.0, 0.0, 0.0, 0.0), eq.state)
    spec = analyze_instantaneous_spectrum(jac)

    traj = engine.simulate(
        initial_state=eq.state, t_span=(0.0, 50.0), dt=0.05,
        drift_fn=drift_fn, diffusion_fn=diff_fn, random_seed=42,
    )

    z_series = ModalProjection.compute_trajectory_projection(traj.states, eq.state, spec.left_critical_vector)
    var_z = compute_rolling_variance(z_series, window_size=50)
    ar1_z = compute_rolling_ar1(z_series, window_size=50)

    is_valid = bool(np.all(var_z >= 0.0) and not np.any(np.isnan(ar1_z)))
    status = "PASSED" if is_valid else "FAILED"

    return TestResult(
        test_id="TEST_11",
        name="EWS Modal Projection Lead Signals",
        status=status,
        details={"mean_variance": float(np.mean(var_z)), "mean_ar1": float(np.mean(ar1_z))},
    )