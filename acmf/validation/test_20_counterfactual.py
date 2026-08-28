import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.decision.causal_mapping import PolicyIntervention, apply_causal_policy_mapping
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_20(params: ModelParameters) -> TestResult:
    """TEST 20 — Контрфактическое моделирование каузальных регуляторных интервенций."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)

    # Кризисное начальное состояние
    init_state = np.zeros(13, dtype=np.float64)
    init_state[0:3] = 1.5
    init_state[3:7] = 0.3

    # 1. Траектория без вмешательства (Baseline)
    def drift_baseline(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), params)

    traj_base = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 15.0),
        dt=0.05,
        drift_fn=drift_baseline,
        diffusion_fn=lambda x: (np.zeros(3), np.zeros(3)),
    )

    # 2. Траектория с регуляторной интервенцией u (Реформа + Наращивание емкости)
    policy = PolicyIntervention(u_reform=0.8, u_capacity=0.5, u_mitigation=0.3)
    policy_params = apply_causal_policy_mapping(params, policy)

    def drift_policy(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(st, forcing, d_a, d_p, d_i, d_agg, np.zeros(3), policy_params)

    traj_policy = engine.simulate(
        initial_state=init_state,
        t_span=(0.0, 15.0),
        dt=0.05,
        drift_fn=drift_policy,
        diffusion_fn=lambda x: (np.zeros(3), np.zeros(3)),
    )

    final_inst_base = traj_base.states[-1, 3]
    final_inst_policy = traj_policy.states[-1, 3]

    # Политика обязана улучшить институциональное состояние относительно baseline
    is_effective = final_inst_policy > final_inst_base
    status = "PASSED" if is_effective else "FAILED"

    return TestResult(
        test_id="TEST_20",
        name="Counterfactual Causal Policy Evaluation",
        status=status,
        details={
            "final_inst_baseline": float(final_inst_base),
            "final_inst_policy": float(final_inst_policy),
        },
    )