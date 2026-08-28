import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.analysis.basins import BasinClassifier
from acmf.validation.result import TestResult


def run_test_10(params: ModelParameters) -> TestResult:
    """TEST 10 — Классификация каскадного гибридного коллапса (Hybrid Cascade Regime)."""
    healthy_target = np.zeros(13, dtype=np.float64)
    healthy_target[3:7] = 0.8
    healthy_target[7] = 0.8 * params.F_max

    # Состояние мультисистемного коллапса (Институты + Экономика)
    hybrid_state = np.zeros(13, dtype=np.float64)
    hybrid_state[0:2] = params.SID_max - 0.05  # SID^1 и SID^2 на границе коллапса
    hybrid_state[3:7] = 0.05

    regime = BasinClassifier.classify_trajectory(
        trajectory_final_state=hybrid_state,
        healthy_target=healthy_target,
        sid_max=params.SID_max,
        tol=0.2,
    )

    status = "PASSED" if regime == "Hybrid" else "FAILED"

    return TestResult(
        test_id="TEST_10",
        name="Hybrid Cascade Regime Classification",
        status=status,
        details={"detected_regime": regime},
    )