import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.analysis.separatrix import SeparatrixSolver
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_07(params: ModelParameters) -> TestResult:
    """TEST 07 — Определение дистанции до сепаратрисы."""
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
    guess_healthy = np.zeros(state_dim, dtype=np.float64)
    guess_healthy[params.N_sub : params.N_sub + 4] = 0.8
    guess_healthy[params.N_sub + 4] = 0.8 * params.F_max
    eq_healthy = eq_engine.find_equilibrium(lambda x: drift_fn(x, 0.0, 0.0, 0.0, 0.0), guess_healthy)

    def is_healthy(state_to_test: np.ndarray) -> bool:
        projected_state, _ = engine.reflector.reflect_state(state_to_test)
        traj = engine.simulate(
            initial_state=projected_state,
            t_span=(0.0, 20.0),
            dt=0.1,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
        )
        final_inst = traj.states[-1, params.N_sub]
        final_sid = traj.states[-1, 0:params.N_sub]
        return bool(final_inst > 0.4 and np.all(final_sid < 1.8))

    active_indices = list(range(params.N_sub + 4))

    solver = SeparatrixSolver()
    threshold = solver.compute_shock_threshold(
        current_state=eq_healthy.state,
        is_healthy_fn=is_healthy,
        active_indices=active_indices,
        max_search_norm=2.5,
        n_directions=16,
    )

    status = "PASSED" if threshold > 0.1 else "FAILED"

    return TestResult(
        test_id="TEST_07",
        name="Shock Threshold Separatrix Distance",
        status=status,
        details={"shock_threshold_distance": float(threshold)},
    )