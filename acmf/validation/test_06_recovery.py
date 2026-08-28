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
    """TEST 06 — Оценка скорости релаксации (Recovery Rate) после малого импульса."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params
        )

    def diff_fn(x):
        return np.zeros(3), np.zeros(3)

    guess = np.zeros(13, dtype=np.float64)
    guess[3:7] = 0.8
    guess[7] = 0.8 * params.F_max
    eq = eq_engine.find_equilibrium(lambda x: drift_fn(x, 0.0, 0.0, 0.0, 0.0), guess)

    # Старт из точки равновесия с возмущением по Inst (-0.1)
    init_state = eq.state.copy()
    delta_inst = 0.1
    init_state[3] -= delta_inst

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 40.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    eq_inst = eq.state[3]
    initial_gap = abs(traj.states[0, 3] - eq_inst)
    final_gap = abs(traj.states[-1, 3] - eq_inst)

    is_recovered = final_gap < (initial_gap * 0.2)
    status = "PASSED" if is_recovered else "FAILED"

    return TestResult(
        test_id="TEST_06",
        name="System Recovery and Resilience",
        status=status,
        details={"initial_gap": float(initial_gap), "final_gap": float(final_gap)},
    )