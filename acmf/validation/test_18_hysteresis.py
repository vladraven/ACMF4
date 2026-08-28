import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_18(params: ModelParameters) -> TestResult:
    """TEST 18 — Path dependence при одинаковом макросостоянии и разном Scar."""
    domain = SolverDomain(
        sid_buf=params.SID_buf,
        sid_max=params.SID_max,
        f_max=params.F_max,
    )
    engine = ACMFEngine(
        domain=domain,
        scheme="euler_maruyama",
    )
    forcing = ForcingProfile().evaluate(0.0)

    base_state = np.zeros(
        9 + params.N_sub + 1,
        dtype=np.float64,
    )
    base_state[0:params.N_sub] = 2.0
    base_state[3:7] = 0.5
    base_state[8] = 0.0

    scar_state = base_state.copy()
    scar_state[8] = 0.7

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st,
            forcing,
            d_a,
            d_p,
            d_i,
            d_agg,
            np.zeros(params.N_sub),
            params,
        )

    def diff_fn(x):
        return np.zeros(params.N_sub), np.zeros(params.N_sub)

    traj_base = engine.simulate(
        initial_state=base_state,
        t_span=(0.0, 20.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    traj_scar = engine.simulate(
        initial_state=scar_state,
        t_span=(0.0, 20.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    final_distance = float(
        np.linalg.norm(
            traj_base.states[-1]
            - traj_scar.states[-1]
        )
    )

    scar_difference = float(
        abs(
            traj_base.states[-1, 8]
            - traj_scar.states[-1, 8]
        )
    )

    status = (
        "PASSED"
        if final_distance > 0.0
        and scar_difference > 0.0
        else "FAILED"
    )

    return TestResult(
        test_id="TEST_18",
        name="Scar Path Dependence and Hysteresis",
        status=status,
        details={
            "final_state_distance": final_distance,
            "final_scar_difference": scar_difference,
            "scar_initial_base": 0.0,
            "scar_initial_history": 0.7,
        },
    )
