import numpy as np

from acmf.decision.causal_mapping import (
    PolicyIntervention,
    apply_causal_policy_mapping,
)
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.forcing import ForcingProfile
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult


def run_test_20(params: ModelParameters) -> TestResult:
    """TEST 20 — Внутренняя проверка model-implied policy counterfactual."""

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

    def simulate(model_params: ModelParameters):
        engine = ACMFEngine(
            domain=domain,
            scheme="euler_maruyama",
        )

        def drift_fn(
            x: np.ndarray,
            d_a: float,
            d_p: float,
            d_i: float,
            d_agg: float,
        ) -> np.ndarray:
            st = StateVector(x)

            return compute_full_drift_vector(
                st,
                forcing,
                d_a,
                d_p,
                d_i,
                d_agg,
                np.zeros(
                    model_params.N_sub,
                    dtype=np.float64,
                ),
                model_params,
            )

        def diff_fn(
            x: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.zeros(
                    model_params.N_sub,
                    dtype=np.float64,
                ),
                np.zeros(
                    model_params.N_sub,
                    dtype=np.float64,
                ),
            )

        return engine.simulate(
            initial_state=init_state,
            t_span=(0.0, 15.0),
            dt=0.05,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
        )

    def summarize(result) -> tuple[float, float]:
        final_inst = float(result.states[-1, 3])

        final_sid = float(
            np.mean(
                result.states[
                    -1,
                    0:params.N_sub,
                ]
            )
        )

        return final_inst, final_sid

    baseline = simulate(params)

    null_policy = PolicyIntervention()

    mild_policy = PolicyIntervention(
        u_reform=0.4,
        u_capacity=0.25,
        u_mitigation=0.15,
    )

    strong_policy = PolicyIntervention(
        u_reform=0.8,
        u_capacity=0.5,
        u_mitigation=0.3,
    )

    null_params = apply_causal_policy_mapping(
        params,
        null_policy,
    )

    mild_params = apply_causal_policy_mapping(
        params,
        mild_policy,
    )

    strong_params = apply_causal_policy_mapping(
        params,
        strong_policy,
    )

    null_result = simulate(null_params)
    mild_result = simulate(mild_params)
    strong_result = simulate(strong_params)

    baseline_inst, baseline_sid = summarize(
        baseline
    )

    null_inst, null_sid = summarize(
        null_result
    )

    mild_inst, mild_sid = summarize(
        mild_result
    )

    strong_inst, strong_sid = summarize(
        strong_result
    )

    baseline_null_difference = float(
        np.max(
            np.abs(
                baseline.states
                - null_result.states
            )
        )
    )

    null_equivalence_tol = 1e-10

    null_equivalence = (
        baseline_null_difference
        <= null_equivalence_tol
    )

    mild_effect = (
        mild_inst - baseline_inst
    )

    strong_effect = (
        strong_inst - baseline_inst
    )

    nontrivial_policy_effect = (
        strong_effect > 0.0
    )

    dose_response = (
        strong_effect
        >= mild_effect - 1e-10
    )

    sid_nonworsening = (
        strong_sid
        <= baseline_sid + 1e-10
    )

    status = (
        "PASSED"
        if (
            null_equivalence
            and nontrivial_policy_effect
            and dose_response
            and sid_nonworsening
        )
        else "FAILED"
    )

    return TestResult(
        test_id="TEST_20",
        name="Counterfactual Causal Policy Evaluation",
        status=status,
        details={
            "final_inst_baseline": baseline_inst,
            "final_inst_null_policy": null_inst,
            "final_inst_mild_policy": mild_inst,
            "final_inst_strong_policy": strong_inst,
            "final_sid_baseline": baseline_sid,
            "final_sid_null_policy": null_sid,
            "final_sid_mild_policy": mild_sid,
            "final_sid_strong_policy": strong_sid,
            "null_policy_max_difference": (
                baseline_null_difference
            ),
            "null_policy_equivalent": (
                null_equivalence
            ),
            "mild_institutional_effect": (
                mild_effect
            ),
            "strong_institutional_effect": (
                strong_effect
            ),
            "nontrivial_policy_effect": (
                nontrivial_policy_effect
            ),
            "dose_response": dose_response,
            "sid_nonworsening": (
                sid_nonworsening
            ),
        },
    )