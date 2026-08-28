from typing import Callable
import numpy as np


class SeparatrixSolver:
    """
    Численный поиск минимального критического шока до границы бассейна:
    D_separatrix(X) = inf ||delta X||_Sigma при X + delta X not in B_healthy
    """

    def compute_shock_threshold(
        self,
        current_state: np.ndarray,
        is_healthy_fn: Callable[[np.ndarray], bool],
        active_indices: list[int] | None = None,
        max_search_norm: float = 3.0,
        n_directions: int = 16,
    ) -> float:
        """Сканирует лучи возмущений в активном подпространстве фазовых переменных."""
        dim = len(current_state)
        indices = active_indices if active_indices is not None else list(range(dim))

        min_dist = float("inf")
        rng = np.random.default_rng(42)

        for _ in range(n_directions):
            direction = np.zeros(dim, dtype=np.float64)
            sub_dir = rng.standard_normal(len(indices))
            sub_norm = np.linalg.norm(sub_dir)
            if sub_norm < 1e-8:
                continue
            direction[indices] = sub_dir / sub_norm

            # Проверка наличия выхода из бассейна на максимальном радиусе
            perturbed_max = current_state + max_search_norm * direction
            if is_healthy_fn(perturbed_max):
                continue

            # Бинарный поиск границы бассейна вдоль направления
            r_low = 0.0
            r_high = max_search_norm

            for _ in range(24):
                r_mid = 0.5 * (r_low + r_high)
                perturbed = current_state + r_mid * direction
                if is_healthy_fn(perturbed):
                    r_low = r_mid
                else:
                    r_high = r_mid

            dist = float(r_high)
            if dist < min_dist:
                min_dist = dist

        return min_dist if min_dist != float("inf") else max_search_norm