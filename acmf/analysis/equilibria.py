from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class EquilibriumConfig:
    residual_tol: float = 1e-6
    uniqueness_tol: float = 1e-3


@dataclass(frozen=True)
class EquilibriumPoint:
    state: np.ndarray
    residual_norm: float
    is_valid: bool


class EquilibriumEngine:
    def __init__(self, config: EquilibriumConfig | None = None) -> None:
        self.config = config or EquilibriumConfig()

    def find_equilibrium(
        self,
        deterministic_drift_fn: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
    ) -> EquilibriumPoint:
        sol = root(deterministic_drift_fn, initial_guess, method="hybr")
        residual = float(np.linalg.norm(sol.fun))
        is_valid = bool(sol.success and residual < self.config.residual_tol)
        return EquilibriumPoint(state=sol.x, residual_norm=residual, is_valid=is_valid)

    def scan_multistability(
        self,
        deterministic_drift_fn: Callable[[np.ndarray], np.ndarray],
        initial_guesses: list[np.ndarray],
        domain_sampler: Callable[[], np.ndarray] | None = None,
        n_samples: int = 50,
    ) -> list[EquilibriumPoint]:
        """
        Поиск нескольких стационарных ветвей.
        Дополняет переданные guess'ы случайными сэмплами из домена.
        """
        all_guesses = list(initial_guesses)
        
        if domain_sampler is not None and len(all_guesses) < n_samples:
            rng = np.random.default_rng(42)
            dim = len(all_guesses[0]) if all_guesses else 13
            for _ in range(n_samples - len(all_guesses)):
                guess = domain_sampler()
                all_guesses.append(guess)

        unique_eqs: list[EquilibriumPoint] = []
        for guess in all_guesses:
            eq = self.find_equilibrium(deterministic_drift_fn, guess)
            if eq.is_valid:
                if not any(
                    np.linalg.norm(eq.state - u.state) < self.config.uniqueness_tol
                    for u in unique_eqs
                ):
                    unique_eqs.append(eq)
        return unique_eqs