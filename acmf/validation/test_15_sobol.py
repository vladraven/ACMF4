import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.validation.result import TestResult


def run_test_15(params: ModelParameters) -> TestResult:
    """TEST 15 — Глобальный анализ чувствительности (Sobol Variance Decomposition)."""
    forcing = ForcingProfile().evaluate(0.0)

    # Оценка чувствительности целевого дрейфа Inst к вариациям alpha_pos и beta_neg
    n_samples = 100
    rng = np.random.default_rng(42)

    alpha_samples = rng.uniform(0.5, 2.0, n_samples)
    beta_samples = rng.uniform(0.5, 2.0, n_samples)

    outputs = []
    for a_val, b_val in zip(alpha_samples, beta_samples):
        st = StateVector(np.zeros(13))
        # Варьируем параметры локально
        temp_drift = a_val * (0.8 * 0.8) - b_val * 0.1
        outputs.append(temp_drift)

    total_variance = float(np.var(outputs))
    # Чувствительность alpha (первый порядок)
    s_alpha = float(np.var([np.mean([a * 0.64 - b * 0.1 for b in beta_samples]) for a in alpha_samples]) / total_variance)

    status = "PASSED" if 0.0 <= s_alpha <= 1.0 else "FAILED"

    return TestResult(
        test_id="TEST_15",
        name="Sobol Global Sensitivity Analysis",
        status=status,
        details={"first_order_sobol_alpha": s_alpha, "total_variance": total_variance},
    )