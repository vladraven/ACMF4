import numpy as np
from acmf.solver.base import StochasticStepScheme


class MilsteinStep(StochasticStepScheme):
    """
    Схема Мильштейна со скачками и поправкой на производную диффузии:
    Y_{n+1} = X_n + Drift_n * dt + sigma_n * dW_n
              + 0.5 * sigma_n * (d_sigma/d_SID) * (dW_n^2 - dt) + Jump_n
    """

    def step(
        self,
        current_state: np.ndarray,
        drift: np.ndarray,
        diffusion_sigma: np.ndarray,
        diffusion_derivative: np.ndarray,
        random_normal: np.ndarray,
        jump_vector: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        dw = np.sqrt(dt) * random_normal

        # 1-й порядок диффузии (Эйлер)
        diffusion_term = np.zeros(13, dtype=np.float64)
        diffusion_term[0:3] = diffusion_sigma * dw

        # Поправка Мильштейна второго порядка
        milstein_correction = np.zeros(13, dtype=np.float64)
        milstein_correction[0:3] = 0.5 * diffusion_sigma * diffusion_derivative * (dw**2 - dt)

        jump_term = np.zeros(13, dtype=np.float64)
        jump_term[0:3] = jump_vector

        y_next = current_state + drift * dt + diffusion_term + milstein_correction + jump_term
        return y_next