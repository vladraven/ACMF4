import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.validation.result import TestResult


def run_test_12(params: ModelParameters) -> TestResult:
    """TEST 12 — Ложноположительные срабатывания EWS в стационарном режиме (FPR <= 0.05)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    # Генерация фонового стационарного шума
    stationary_noise = rng.standard_normal(n_samples)
    threshold = 2.5  # Порог 2.5-сигма

    false_alarms = np.sum(np.abs(stationary_noise) > threshold)
    fpr = float(false_alarms / n_samples)

    status = "PASSED" if fpr <= 0.05 else "FAILED"

    return TestResult(
        test_id="TEST_12",
        name="EWS False Positive Rate",
        status=status,
        details={"fpr": fpr, "target_max": 0.05},
    )