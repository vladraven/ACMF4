import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_01(params: ModelParameters) -> TestResult:
    """TEST 01 — Отраженная диффузия Скорохода: проверка компенсатора и сохранения домена Omega."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="milstein")
    forcing = ForcingProfile().evaluate(0.0)

    # Старт вблизи границы
    init_state = np.zeros(13, dtype=np.float64)
    init_state[0:3] = params.SID_max * 0.95

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params
        )

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    # Генератор импульсных возмущений за пределы границы для проверки проектора Скорохода
    def jump_generator(x, t):
        if 1.0 <= t <= 1.05:
            return np.array([2.0, 2.0, 2.0], dtype=np.float64)
        if 3.0 <= t <= 3.05:
            return np.array([-5.0, -5.0, -5.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    traj = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 5.0),
        dt=0.01,
        drift_fn=drift_fn,
        diffusion_fn=diff_fn,
        jump_generator_fn=jump_generator,
        random_seed=123,
    )

    all_in_domain = all(StateVector(s).is_in_domain(params) for s in traj.states)
    reflections_count = sum(1 for d in traj.diagnostics if d.reflected)

    status = "PASSED" if all_in_domain and reflections_count > 0 else "FAILED"

    return TestResult(
        test_id="TEST_01",
        name="Skorokhod Domain Invariance",
        status=status,
        details={"reflections_count": reflections_count, "domain_preserved": all_in_domain},
    )