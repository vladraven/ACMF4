import numpy as np
from acmf.solver.base import StochasticStepScheme


class EulerMaruyamaStep(StochasticStepScheme):
    """
    Схема Эйлера–Маруямы со скачками:
    Y_{n+1} = X_n + Drift_n * dt + sigma_n * dW_n + Jump_n
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
        # Диффузия действует только на SID [0:3]
        diffusion_term = np.zeros(13, dtype=np.float64)
        diffusion_term[0:3] = diffusion_sigma * dw

        jump_term = np.zeros(13, dtype=np.float64)
        jump_term[0:3] = jump_vector

        y_next = current_state + drift * dt + diffusion_term + jump_term
        return y_next