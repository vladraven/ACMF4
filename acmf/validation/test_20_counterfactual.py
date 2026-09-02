import numpy as np
from acmf.decision.causal_mapping import PolicyIntervention, apply_causal_policy_mapping
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.forcing import ForcingProfile
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_20(params: ModelParameters) -> TestResult:
    """TEST 20 — Внутренняя проверка model-implied policy counterfactual."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    forcing = ForcingProfile().evaluate(0.0)

    state_dim = 2 * params.N_sub + 7
    init_state = np.zeros(state_dim, dtype=np.float64)
    init_state[0:params.N_sub] = 1.5
    init_state[params.N_sub : params.N_sub + 4] = 0.3

    def simulate(model_params: ModelParameters):
        engine = ACMFEngine(domain=domain, scheme="euler_maruyama", state_dim=state_dim, sid_noise_dim=model_params.N_sub)

        def drift_fn(x, d_a, d_p, d_i, d_agg):
            st = StateVector(x)
            return compute_full_drift_vector(st, forcing, d_a, d_p, d_i, d_agg, np.zeros(model_params.N_sub), model_params)

        def diff_fn(x):
            return np.zeros(model_params.N_sub), np.zeros(model_params.N_sub)

        return engine.simulate(
            initial_state=init_state, t_span=(0.0, 15.0), dt=0.05,
            drift_fn=drift_fn, diffusion_fn=diff_fn,
        )

    def summarize(result):
        final_inst = float(result.states[-1, params.N_sub])
        final_sid = float(np.mean(result.states[-1, 0:params.N_sub]))
        return final_inst, final_sid

    baseline = simulate(params)
    null_policy = PolicyIntervention()
    mild_policy = PolicyIntervention(u_reform=0.4, u_capacity=0.25, u_mitigation=0.15)
    strong_policy = PolicyIntervention(u_reform=0.8, u_capacity=0.5, u_mitigation=0.3)

    null_result = simulate(apply_causal_policy_mapping(params, null_policy))
    mild_result = simulate(apply_causal_policy_mapping(params, mild_policy))
    strong_result = simulate(apply_causal_policy_mapping(params, strong_policy))

    baseline_inst, baseline_sid = summarize(baseline)
    null_inst, null_sid = summarize(null_result)
    mild_inst, mild_sid = summarize(mild_result)
    strong_inst, strong_sid = summarize(strong_result)

    baseline_null_diff = float(np.max(np.abs(baseline.states - null_result.states)))
    null_equivalence = baseline_null_diff <= 1e-10

    nontrivial = (strong_inst - baseline_inst) > 0.0
    dose_response = (strong_inst - baseline_inst) >= (mild_inst - baseline_inst) - 1e-10
    sid_nonworsening = strong_sid <= baseline_sid + 1e-10

    status = "PASSED" if (null_equivalence and nontrivial and dose_response and sid_nonworsening) else "FAILED"

    return TestResult(
        test_id="TEST_20",
        name="Counterfactual Causal Policy Evaluation",
        status=status,
        details={
            "final_inst_baseline": baseline_inst,
            "final_inst_policy": strong_inst,
        },
    )