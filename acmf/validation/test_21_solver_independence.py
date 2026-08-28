import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_21(params: ModelParameters) -> TestResult:
    """TEST 21 — Сходимость и независимость Euler-Maruyama/Milstein."""
    domain = SolverDomain(
        sid_buf=params.SID_buf,
        sid_max=params.SID_max,
        f_max=params.F_max,
    )
    forcing = ForcingProfile().evaluate(0.0)

    init_state = np.zeros(
        9 + params.N_sub + 1,
        dtype=np.float64,
    )
    init_state[3:7] = 0.5

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
        st = StateVector(x)
        return compute_diffusion_sigma(
            st,
            forcing,
            params,
        )

    dt_values = (0.04, 0.02, 0.01)
    discrepancies = []

    for dt in dt_values:
        engine_em = ACMFEngine(
            domain=domain,
            scheme="euler_maruyama",
        )
        engine_mil = ACMFEngine(
            domain=domain,
            scheme="milstein",
        )

        traj_em = engine_em.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=42,
        )
        traj_mil = engine_mil.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=42,
        )

        if traj_em.states.shape != traj_mil.states.shape:
            return TestResult(
                test_id="TEST_21",
                name="Solver Independence Convergence",
                status="FAILED",
                details={"reason": "trajectory_shape_mismatch"},
            )

        discrepancies.append(
            float(
                np.max(
                    np.abs(
                        traj_em.states
                        - traj_mil.states
                    )
                )
            )
        )

    converges = (
        discrepancies[-1]
        <= discrepancies[0]
    )

    return TestResult(
        test_id="TEST_21",
        name="Solver Independence Convergence",
        status="PASSED" if converges else "FAILED",
        details={
            "dt_values": dt_values,
            "max_discrepancies": discrepancies,
            "refinement_improves_agreement": converges,
        },
    )
