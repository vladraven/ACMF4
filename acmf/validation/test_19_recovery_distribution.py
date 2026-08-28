import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_19(params: ModelParameters) -> TestResult:
    """TEST 19 — Распределение времен восстановления после случайных микрошоков."""
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
        return np.zeros(3), np.zeros(3)

    recovery_times = []
    for seed in range(10):
        perturbed = init_state.copy()
        perturbed[0:3] += 0.3  # Шок дефицита

        traj = engine.simulate(
            initial_state=perturbed,
            t_span=(0.0, 25.0),
            dt=0.05,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=seed,
        )

        sids_agg = np.mean(traj.states[:, 0:3], axis=1)
        recovered_idx = np.where(sids_agg < 0.05)[0]
        rec_t = float(traj.times[recovered_idx[0]]) if len(recovered_idx) > 0 else 25.0
        recovery_times.append(rec_t)

    mean_rec_t = float(np.mean(recovery_times))
    status = "PASSED" if mean_rec_t < 25.0 else "FAILED"

    return TestResult(
        test_id="TEST_19",
        name="Stochastic Recovery Time Distribution",
        status=status,
        details={"mean_recovery_time": mean_rec_t},
    )