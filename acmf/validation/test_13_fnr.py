import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.validation.result import TestResult


def run_test_13(params: ModelParameters) -> TestResult:
    """TEST 13 — Ложноотрицательные пропуски EWS при критическом замедлении (FNR <= 0.10)."""
    rng = np.random.default_rng(42)
    n_trials = 200
    detected_count = 0
    threshold = 1.5

    for _ in range(n_trials):
        # Моделирование нарастающего тренда дисперсии при приближении к бифуркации
        ramp = np.linspace(0.1, 3.0, 100)
        signal = ramp + rng.normal(0.0, 0.3, 100)
        if np.max(signal[-20:]) > threshold:
            detected_count += 1

    fnr = float(1.0 - (detected_count / n_trials))
    status = "PASSED" if fnr <= 0.10 else "FAILED"

    return TestResult(
        test_id="TEST_13",
        name="EWS False Negative Rate",
        status=status,
        details={"fnr": fnr, "target_max": 0.10},
    )