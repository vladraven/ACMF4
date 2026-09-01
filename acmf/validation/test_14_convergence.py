import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult

SAMPLE_RUNS_COUNT: int = 50
SIMULATION_HORIZON: float = 10.0
INTEGRATION_STEP: float = 0.05
CI_RELATIVE_LIMIT: float = 0.05
ABSOLUTE_DIFF_LIMIT: float = 1e-3


def run_test_14(params: ModelParameters) -> TestResult:
    """TEST 14 — Монте-Карло сходимость статистических моментов траектории."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    state_dim = 2 * params.N_sub + 7
    engine = ACMFEngine(
        domain=domain,
        scheme="euler_maruyama",
        state_dim=state_dim,
        sid_noise_dim=params.N_sub,
    )
    forcing = ForcingProfile().evaluate(0.0)

    init_state = np.zeros(state_dim, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(params.N_sub), params
        )

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    final_inst_values = []

    for seed in range(SAMPLE_RUNS_COUNT):
        traj = engine.simulate(
            initial_state=init_state,
            t_span=(0.0, SIMULATION_HORIZON),
            dt=INTEGRATION_STEP,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=seed,
        )
        final_inst_values.append(traj.states[-1, params.N_sub])

    mean_val = float(np.mean(final_inst_values))
    std_err = float(np.std(final_inst_values, ddof=1) / np.sqrt(SAMPLE_RUNS_COUNT))

    ci_halfwidth = 1.96 * std_err
    domain_range = 1.0
    ci_relative = ci_halfwidth / domain_range

    half = SAMPLE_RUNS_COUNT // 2
    mean_first = float(np.mean(final_inst_values[:half]))
    mean_second = float(np.mean(final_inst_values[half:]))
    pooled_se = float(
        np.sqrt(
            (np.std(final_inst_values[:half], ddof=1) ** 2 / half)
            + (np.std(final_inst_values[half:], ddof=1) ** 2 / half)
        )
    )
    diff_means = abs(mean_first - mean_second)
    half_agreement = (diff_means < 1.96 * pooled_se) or (diff_means < ABSOLUTE_DIFF_LIMIT)

    is_converged = (ci_relative < CI_RELATIVE_LIMIT) and half_agreement

    status = "PASSED" if is_converged else "FAILED"

    return TestResult(
        test_id="TEST_14",
        name="Monte Carlo Moment Convergence",
        status=status,
        details={
            "mean_inst": mean_val,
            "std_error": std_err,
            "ci_halfwidth": ci_halfwidth,
            "ci_relative_to_domain": ci_relative,
            "mean_first_half": mean_first,
            "mean_second_half": mean_second,
            "half_agreement": half_agreement,
        },
    )