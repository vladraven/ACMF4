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
    """TEST 20 — Контрфактическая оценка каузальной интервенции."""
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
    init_state[0:params.N_sub] = 1.5
    init_state[3:7] = 0.3

    policy = PolicyIntervention(
        u_reform=0.8,
        u_capacity=0.5,
        u_mitigation=0.3,
    )
    policy_params = apply_causal_policy_mapping(
        params,
        policy,
    )

    def simulate(model_params):
        engine = ACMFEngine(
            domain=domain,
            scheme="euler_maruyama",
        )

        def drift_fn(x, d_a, d_p, d_i, d_agg):
            st = StateVector(x)
            return compute_full_drift_vector(
                st,
                forcing,
                d_a,
                d_p,
                d_i,
                d_agg,
                np.zeros(model_params.N_sub),
                model_params,
            )

        def diff_fn(x):
            return np.zeros(model_params.N_sub), np.zeros(model_params.N_sub)

        return engine.simulate(
            initial_state=init_state,
            t_span=(0.0, 15.0),
            dt=0.05,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
        )

    baseline = simulate(params)
    intervention = simulate(policy_params)

    baseline_inst = float(baseline.states[-1, 3])
    intervention_inst = float(intervention.states[-1, 3])
    baseline_sid = float(np.mean(baseline.states[-1, 0:params.N_sub]))
    intervention_sid = float(np.mean(intervention.states[-1, 0:params.N_sub]))

    effect = intervention_inst - baseline_inst
    status = "PASSED" if effect > 0.0 else "FAILED"

    return TestResult(
        test_id="TEST_20",
        name="Counterfactual Causal Policy Evaluation",
        status=status,
        details={
            "final_inst_baseline": baseline_inst,
            "final_inst_policy": intervention_inst,
            "institutional_effect": effect,
            "final_sid_baseline": baseline_sid,
            "final_sid_policy": intervention_sid,
        },
    )
