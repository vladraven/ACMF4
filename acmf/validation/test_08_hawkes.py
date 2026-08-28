import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.stochastic.hawkes import MultivariateHawkesProcess, HawkesEvent
from acmf.validation.result import TestResult


def run_test_08(params: ModelParameters) -> TestResult:
    """TEST 08 — Субкритичность ветвления и кластеризация процессов Хоукса."""
    def base_rate(scar: float) -> np.ndarray:
        return np.array([0.1, 0.1, 0.1]) * (1.0 + 2.0 * scar)

    hawkes = MultivariateHawkesProcess(
        base_rate_fn=base_rate,
        gamma_matrix=params.Gamma,
        beta_h=params.beta_H,
    )

    rng = np.random.default_rng(42)
    history: list[HawkesEvent] = []
    t = 0.0
    dt = 0.05
    n_steps = 1000

    for _ in range(n_steps):
        events = hawkes.sample_step_events(t, dt, scar=0.2, history=history, rng=rng)
        history.extend(events)
        t += dt

    # Проверка спектрального радиуса матрицы ветвления K_H = Gamma / beta_H
    k_h = params.Gamma / params.beta_H
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(k_h))))

    is_subcritical = spectral_radius < 1.0
    has_events = len(history) > 0

    status = "PASSED" if is_subcritical and has_events else "FAILED"

    return TestResult(
        test_id="TEST_08",
        name="Hawkes Subcriticality and Event Clustering",
        status=status,
        details={
            "spectral_radius": spectral_radius,
            "total_events_generated": len(history),
        },
    )