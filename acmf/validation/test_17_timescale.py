import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.dynamics import compute_tsm
from acmf.validation.result import TestResult


def run_test_17(params: ModelParameters) -> TestResult:
    """TEST 17 — Анализ Time-Scale Mismatch (TSM) при ускорении темпов изменений."""
    rates = np.linspace(0.0, 10.0, 100)
    tsm_values = [compute_tsm(r, r, r, params) for r in rates]

    # TSM обязан строго лежать в [0, 1) и быть монотонно возрастающим
    in_bounds = all(0.0 <= val < 1.0 for val in tsm_values)
    is_monotonic = all(x <= y for x, y in zip(tsm_values[:-1], tsm_values[1:]))

    status = "PASSED" if in_bounds and is_monotonic else "FAILED"

    return TestResult(
        test_id="TEST_17",
        name="Time-Scale Mismatch Monotonicity",
        status=status,
        details={"tsm_min": float(tsm_values[0]), "tsm_max": float(tsm_values[-1])},
    )