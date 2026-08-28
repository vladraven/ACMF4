from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class HawkesEvent:
    """Точечное событие многомерного процесса Хоукса."""
    time: float
    dimension: int
    magnitude: np.ndarray


class MultivariateHawkesProcess:
    """
    Многомерный самовозбуждающийся процесс Хоукса:
    lambda_i(t) = lambda_0,i(Scar) + sum_j sum_{t_m < t} Gamma_{ij} * exp(-beta_H * (t - t_m))
    """

    def __init__(
        self,
        base_rate_fn: Callable[[float], np.ndarray],
        gamma_matrix: np.ndarray,
        beta_h: float,
    ) -> None:
        self.base_rate_fn = base_rate_fn
        self.gamma_matrix = np.asarray(gamma_matrix, dtype=np.float64)
        self.beta_h = float(beta_h)
        self.dim = self.gamma_matrix.shape[0]

        # Проверка спектрального радиуса K_H = Gamma / beta_H < 1
        k_h = self.gamma_matrix / self.beta_h
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(k_h))))
        if spectral_radius >= 1.0:
            raise ValueError(f"Нарушено условие субкритичности: rho(K_H) = {spectral_radius:.4f} >= 1.0")

    def compute_intensities(self, t: float, scar: float, history: list[HawkesEvent]) -> np.ndarray:
        """Вычисляет вектор условных интенсивностей lambda(t) формы (dim,)."""
        base = np.asarray(self.base_rate_fn(scar), dtype=np.float64)
        intensities = base.copy()

        for event in history:
            if event.time >= t:
                continue
            dt = t - event.time
            kernel_decay = np.exp(-self.beta_h * dt)
            intensities += self.gamma_matrix[:, event.dimension] * kernel_decay

        return np.maximum(0.0, intensities)

    def sample_step_events(
        self,
        t: float,
        dt: float,
        scar: float,
        history: list[HawkesEvent],
        rng: np.random.Generator,
    ) -> list[HawkesEvent]:
        """Генерирует события Хоукса на шаге [t, t + dt] методом Пуассоновского прореживания (Огата)."""
        intensities = self.compute_intensities(t, scar, history)
        step_events: list[HawkesEvent] = []

        for d in range(self.dim):
            rate = intensities[d]
            prob = 1.0 - np.exp(-rate * dt)
            if rng.uniform(0.0, 1.0) < prob:
                # Величина базового скачка
                raw_mag = rng.standard_exponential(self.dim)
                step_events.append(HawkesEvent(time=t + dt, dimension=d, magnitude=raw_mag))

        return step_events