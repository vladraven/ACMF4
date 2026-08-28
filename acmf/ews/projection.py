import numpy as np


class ModalProjection:
    """Проекция возмущений состояния на доминирующую критическую левую моду Якобиана."""

    @staticmethod
    def project_state(
        state: np.ndarray,
        equilibrium_state: np.ndarray,
        left_critical_vector: np.ndarray,
    ) -> float:
        """Z(t) = Re( w_crit^H * (X(t) - X_eq) )"""
        delta_x = state - equilibrium_state
        return float(np.real(np.dot(np.conj(left_critical_vector), delta_x)))

    @staticmethod
    def compute_trajectory_projection(
        states: np.ndarray,
        equilibrium_state: np.ndarray,
        left_critical_vector: np.ndarray,
    ) -> np.ndarray:
        """Векторизованный расчет скалярного ряда Z(t) для всей траектории."""
        delta = states - equilibrium_state
        # Проекция вдоль оси размерности (N_steps, 13)
        return np.real(delta @ np.conj(left_critical_vector))