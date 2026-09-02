import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_06(params: ModelParameters) -> TestResult:
    """TEST 06 — Оценка скорости релаксации после малого импульса."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama", state_dim=2 * params.N_sub + 7)
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(params.N_sub), params
        )

    def diff_fn(x):
        return np.zeros(params.N_sub), np.zeros(params.N_sub)

    state_dim = 2 * params.N_sub + 7
    guess = np.zeros(state_dim, dtype=np.float64)
    guess[params.N_sub : params.N_sub + 4] = 0.8
    guess[params.N_sub + 4] = 0.8 * params.F_max
    eq = eq_engine.find_equilibrium(lambda x: drift_fn(x, 0.0, 0.0, 0.0, 0.0), guess)

    init_state = eq.state.copy()
    delta_inst = 0.1
    init_state[params.N_sub] -= delta_inst

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 40.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    eq_inst = eq.state[params.N_sub]
    initial_gap = abs(traj.states[0, params.N_sub] - eq_inst)
    final_gap = abs(traj.states[-1, params.N_sub] - eq_inst)

    is_recovered = final_gap < (initial_gap * 0.2)
    status = "PASSED" if is_recovered else "FAILED"

    return TestResult(
        test_id="TEST_06",
        name="System Recovery and Resilience",
        status=status,
        details={"initial_gap": float(initial_gap), "final_gap": float(final_gap)},
    )