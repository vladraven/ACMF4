import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_00(params: ModelParameters) -> TestResult:
    """TEST 00 — Детерминированный базовый сценарий: устойчивость и инвариантность."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)

    init_state = np.zeros(13, dtype=np.float64)
    init_state[3:7] = 0.8  # Inst, Ch, Prod, M
    init_state[7] = 0.8 * params.F_max

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params
        )

    def diff_fn(x):
        return np.zeros(3), np.zeros(3)

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 50.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    final_st = StateVector(traj.states[-1])
    in_domain = final_st.is_in_domain(params)
    final_drift_norm = float(np.linalg.norm(traj.drifts[-1]))

    status = "PASSED" if in_domain and not np.isnan(final_drift_norm) else "FAILED"

    return TestResult(
        test_id="TEST_00",
        name="Deterministic Baseline Convergence",
        status=status,
        details={"final_drift_norm": final_drift_norm, "in_domain": in_domain},
    )