from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class EquilibriumPoint:
    """Точка равновесия детерминированной системы."""
    state: np.ndarray
    residual_norm: float
    is_valid: bool


class EquilibriumEngine:
    """Поиск и валидация стационарных состояний F(X*) = 0."""

    def __init__(self, epsilon_eq: float = 1e-6) -> None:
        self.epsilon_eq = epsilon_eq

    def find_equilibrium(
        self,
        deterministic_drift_fn: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
    ) -> EquilibriumPoint:
        """Находит точку равновесия методом гибридного алгоритма Пауэлла (MINPACK)."""
        sol = root(deterministic_drift_fn, initial_guess, method="hybr")
        residual = float(np.linalg.norm(sol.fun))
        is_valid = bool(sol.success and residual < self.epsilon_eq)

        return EquilibriumPoint(
            state=sol.x,
            residual_norm=residual,
            is_valid=is_valid,
        )

    def scan_multistability(
        self,
        deterministic_drift_fn: Callable[[np.ndarray], np.ndarray],
        initial_guesses: list[np.ndarray],
    ) -> list[EquilibriumPoint]:
        """Поиск нескольких стационарных ветвей из различных начальных приближений."""
        unique_eqs: list[EquilibriumPoint] = []
        for guess in initial_guesses:
            eq = self.find_equilibrium(deterministic_drift_fn, guess)
            if eq.is_valid:
                if not any(np.linalg.norm(eq.state - u.state) < 1e-3 for u in unique_eqs):
                    unique_eqs.append(eq)
        return unique_eqs