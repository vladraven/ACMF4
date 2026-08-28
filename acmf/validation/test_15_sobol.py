import numpy as np
from dataclasses import replace
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.validation.result import TestResult


def run_test_15(params: ModelParameters) -> TestResult:
    """TEST 15 — Глобальный анализ чувствительности реального ACMF drift."""
    forcing = ForcingProfile().evaluate(0.0)
    rng = np.random.default_rng(42)
    n_samples = 100

    raw_state = np.zeros(9 + params.N_sub + 1, dtype=np.float64)
    raw_state[0:params.N_sub] = 0.5
    raw_state[3:7] = 0.8
    raw_state[7] = 0.8 * params.F_max
    state = StateVector(raw_state)

    alpha_samples = rng.uniform(0.5, 2.0, n_samples)
    beta_samples = rng.uniform(0.5, 2.0, n_samples)

    outputs = np.empty(n_samples, dtype=np.float64)

    for index, (alpha, beta) in enumerate(
        zip(alpha_samples, beta_samples)
    ):
        sample_params = replace(
            params,
            alpha_pos=float(alpha),
            beta_neg=float(beta),
        )
        drift = compute_full_drift_vector(
            state,
            forcing,
            0.0,
            0.0,
            0.0,
            0.0,
            np.zeros(params.N_sub),
            sample_params,
        )
        outputs[index] = drift[3]

    total_variance = float(np.var(outputs, ddof=1))

    if total_variance <= np.finfo(float).eps:
        return TestResult(
            test_id="TEST_15",
            name="Sobol Global Sensitivity Analysis",
            status="FAILED",
            details={
                "reason": "zero_output_variance",
                "total_variance": total_variance,
            },
        )

    conditional_means = np.empty(n_samples, dtype=np.float64)

    for index, alpha in enumerate(alpha_samples):
        values = []
        for beta in beta_samples:
            sample_params = replace(
                params,
                alpha_pos=float(alpha),
                beta_neg=float(beta),
            )
            drift = compute_full_drift_vector(
                state,
                forcing,
                0.0,
                0.0,
                0.0,
                0.0,
                np.zeros(params.N_sub),
                sample_params,
            )
            values.append(drift[3])
        conditional_means[index] = float(np.mean(values))

    first_order_alpha = float(
        np.var(conditional_means, ddof=1)
        / total_variance
    )

    status = (
        "PASSED"
        if np.isfinite(first_order_alpha)
        and 0.0 <= first_order_alpha <= 1.0
        else "FAILED"
    )

    return TestResult(
        test_id="TEST_15",
        name="Sobol Global Sensitivity Analysis",
        status=status,
        details={
            "first_order_sobol_alpha": first_order_alpha,
            "total_variance": total_variance,
            "n_samples": n_samples,
        },
    )
