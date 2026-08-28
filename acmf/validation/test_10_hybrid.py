import numpy as np

from acmf.analysis.basins import BasinClassifier
from acmf.model.parameters import ModelParameters
from acmf.validation.result import TestResult


def run_test_10(params: ModelParameters) -> TestResult:
    """TEST 10 — Проверка классификации режимов бассейнов и Hybrid Cascade."""

    state_size = 9 + params.N_sub + 1
    tol = 0.2

    healthy_target = np.zeros(
        state_size,
        dtype=np.float64,
    )

    healthy_target[3:7] = 0.8
    healthy_target[7] = 0.8 * params.F_max

    healthy_state = healthy_target.copy()

    anarchy_state = healthy_target.copy()
    anarchy_state[0] = params.SID_max

    default_state = healthy_target.copy()
    default_state[1] = params.SID_max

    depopulation_state = healthy_target.copy()
    depopulation_state[2] = params.SID_max

    hybrid_state = healthy_target.copy()
    hybrid_state[0] = params.SID_max
    hybrid_state[1] = params.SID_max

    intermediate_state = healthy_target.copy()
    intermediate_state[0] = 0.5 * params.SID_max
    intermediate_state[1] = 0.5 * params.SID_max
    intermediate_state[3:7] = 0.5

    cases = {
        "Healthy": healthy_state,
        "Anarchy": anarchy_state,
        "Default": default_state,
        "Depopulation": depopulation_state,
        "Hybrid": hybrid_state,
        "Intermediate": intermediate_state,
    }

    detected: dict[str, str] = {}

    for expected, state in cases.items():
        detected[expected] = (
            BasinClassifier.classify_trajectory(
                trajectory_final_state=state,
                healthy_target=healthy_target,
                sid_max=params.SID_max,
                tol=tol,
            )
        )

    passed_cases = [
        expected
        for expected, actual in detected.items()
        if actual == expected
    ]

    hybrid_state_permuted = healthy_target.copy()
    hybrid_state_permuted[1] = params.SID_max
    hybrid_state_permuted[2] = params.SID_max

    hybrid_permutation_result = (
        BasinClassifier.classify_trajectory(
            trajectory_final_state=hybrid_state_permuted,
            healthy_target=healthy_target,
            sid_max=params.SID_max,
            tol=tol,
        )
    )

    expected_total = len(cases) + 1
    passed_total = len(passed_cases) + int(
        hybrid_permutation_result == "Hybrid"
    )

    status = (
        "PASSED"
        if passed_total == expected_total
        else "FAILED"
    )

    return TestResult(
        test_id="TEST_10",
        name="Hybrid Cascade Regime Classification",
        status=status,
        details={
            "expected_cases": expected_total,
            "passed_cases": passed_total,
            "classification_results": detected,
            "hybrid_permutation_result": (
                hybrid_permutation_result
            ),
        },
    )