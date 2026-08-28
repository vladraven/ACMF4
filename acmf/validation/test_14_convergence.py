import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_14(params: ModelParameters) -> TestResult:
    """TEST 14 — Монте-Карло сходимость статистических моментов траектории."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)

    init_state = np.zeros(13, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params
        )

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    n_runs = 20
    final_inst_values = []

    for seed in range(n_runs):
        traj = engine.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=0.05,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=seed,
        )
        final_inst_values.append(traj.states[-1, 3])

    std_err = float(np.std(final_inst_values) / np.sqrt(n_runs))
    is_converged = std_err < 0.1

    status = "PASSED" if is_converged else "FAILED"

    return TestResult(
        test_id="TEST_14",
        name="Monte Carlo Moment Convergence",
        status=status,
        details={"mean_inst": float(np.mean(final_inst_values)), "std_error": std_err},
    )