import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_00(params: ModelParameters) -> TestResult:
    """TEST 00 — Детерминированный baseline: сходимость к равновесию."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama", state_dim=2 * params.N_sub + 7)
    forcing = ForcingProfile().evaluate(0.0)

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(params.N_sub), params
        )

    def diff_fn(x):
        return np.zeros(params.N_sub), np.zeros(params.N_sub)

    init_state = np.zeros(2 * params.N_sub + 7, dtype=np.float64)
    init_state[params.N_sub : params.N_sub + 4] = 0.8
    init_state[params.N_sub + 4] = 0.8 * params.F_max

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 20.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    final_drift = drift_fn(traj.states[-1], 0.0, 0.0, 0.0, 0.0)
    final_drift_norm = float(np.linalg.norm(final_drift))

    # Строгая проверка: drift_norm должен быть малым, а не просто конечным
    is_converged = (not np.isnan(final_drift_norm)) and (final_drift_norm < 1e-4)
    status = "PASSED" if is_converged else "FAILED"

    return TestResult(
        test_id="TEST_00",
        name="Deterministic Baseline Convergence",
        status=status,
        details={"final_drift_norm": final_drift_norm, "in_domain": True},
    )