import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState
from acmf.model.metabolism import compute_order_generation, compute_r_eff
from acmf.validation.result import TestResult


def run_test_16(params: ModelParameters) -> TestResult:
    """TEST 16 — Допустимость метаболического порядка и ECY: 0 <= ECY^k < Capacity^k."""
    rng = np.random.default_rng(42)
    n_samples = 500
    all_valid = True

    max_ratio = 0.0

    for _ in range(n_samples):
        raw_state = np.zeros(13, dtype=np.float64)
        raw_state[3:7] = rng.uniform(0.0, 1.0, 4)
        raw_state[7] = rng.uniform(0.0, params.F_max)
        raw_state[8] = rng.uniform(0.0, 1.0)
        raw_state[12] = rng.exponential(scale=2.0)  # RecDebt >= 0

        state = StateVector(raw_state)
        forcing = ForcingState(
            A=rng.uniform(0.0, 2.0),
            R_0=rng.uniform(0.0, 2.0),
            V=rng.uniform(0.0, 1.0),
            G=rng.uniform(0.0, 1.0),
        )

        r_eff = compute_r_eff(forcing.R_0, state.scar, params.gamma_R)
        q, ecy = compute_order_generation(state, forcing, r_eff, params)

        if np.any(ecy < 0.0) or np.any(ecy >= params.Capacity):
            all_valid = False
            break

        current_ratio = float(np.max(ecy / params.Capacity))
        if current_ratio > max_ratio:
            max_ratio = current_ratio

    status = "PASSED" if all_valid and max_ratio < 1.0 else "FAILED"

    return TestResult(
        test_id="TEST_16",
        name="Order Generation and ECY Feasibility",
        status=status,
        details={"max_ecy_to_capacity_ratio": max_ratio, "n_samples": n_samples},
    )