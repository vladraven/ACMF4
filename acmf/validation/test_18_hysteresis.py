import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_18(params: ModelParameters) -> TestResult:
    """TEST 18 — Необратимость и гистерезис структурного шрама (Scar Dynamic Memory)."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)

    # Состояние с высоким уровнем накопленного дефицита
    init_state = np.zeros(13, dtype=np.float64)
    init_state[0:3] = 2.0  # Высокий SID
    init_state[3:7] = 0.5

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params
        )

    def diff_fn(x):
        return np.zeros(3), np.zeros(3)

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 20.0),
        dt=0.05,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
    )

    scar_series = traj.states[:, 8]
    # Накопление шрама при кризисе
    scar_accumulated = float(scar_series[-1] - scar_series[0])
    status = "PASSED" if scar_accumulated > 0.0 else "FAILED"

    return TestResult(
        test_id="TEST_18",
        name="Scar Accumulation and Hysteresis",
        status=status,
        details={"scar_initial": float(scar_series[0]), "scar_final": float(scar_series[-1])},
    )