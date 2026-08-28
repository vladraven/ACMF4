import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_19(params: ModelParameters) -> TestResult:
    """TEST 19 — Распределение времени восстановления после стохастических шоков."""
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
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max
    init_state[0:params.N_sub] = 0.3

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

    recovery_times = []
    n_trials = 10
    horizon = 25.0
    dt = 0.05
    recovery_threshold = 0.05

    for seed in range(n_trials):
        perturbed = init_state.copy()
        perturbed[0:params.N_sub] += 0.3

        engine = ACMFEngine(
            domain=domain,
            scheme="euler_maruyama",
        )

        trajectory = engine.simulate(
            initial_state=perturbed,
            t_span=(0.0, horizon),
            dt=dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=seed,
        )

        sids_agg = np.mean(
            trajectory.states[:, 0:params.N_sub],
            axis=1,
        )
        recovered = np.flatnonzero(
            sids_agg < recovery_threshold
        )

        if recovered.size == 0:
            recovery_times.append(np.inf)
        else:
            recovery_times.append(
                float(trajectory.times[recovered[0]])
            )

    finite_times = np.asarray(
        [value for value in recovery_times if np.isfinite(value)],
        dtype=np.float64,
    )

    status = (
        "PASSED"
        if finite_times.size > 0
        else "NOT_DETECTED"
    )

    return TestResult(
        test_id="TEST_19",
        name="Stochastic Recovery Time Distribution",
        status=status,
        details={
            "n_trials": n_trials,
            "recovered_trials": int(finite_times.size),
            "mean_recovery_time": (
                float(np.mean(finite_times))
                if finite_times.size
                else None
            ),
            "recovery_times": recovery_times,
        },
    )
