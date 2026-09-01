from typing import Callable
import numpy as np


DEFAULT_FINITE_DIFFERENCE_STEP: float = 1e-6


def _scaled_step(x_j: float, eps: float) -> float:
    """Масштабированный шаг: eps_j = eps * max(1, |x_j|)."""
    return eps * max(1.0, abs(x_j))


def compute_finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = DEFAULT_FINITE_DIFFERENCE_STEP,
) -> np.ndarray:
    """
    Вычисляет Якобиан вектор-функции F(X) центральными конечными разностями
    с масштабированным шагом.
    """
    dim = len(x)
    f0 = f(x)
    dim_out = len(f0)
    jac = np.zeros((dim_out, dim), dtype=np.float64)

    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        h = _scaled_step(x[j], eps)
        dx[j] = h
        f_plus = f(x + dx)
        f_minus = f(x - dx)
        jac[:, j] = (f_plus - f_minus) / (2.0 * h)

    return jac


def compute_dde_jacobians(
    full_f_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_eq: np.ndarray,
    eps: float = DEFAULT_FINITE_DIFFERENCE_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет матрицы линеаризации DDE системы в точке равновесия:
    dx/dt = A_0 * x(t) + A_1 * x(t - Delta_t)
    """
    dim = len(x_eq)
    a_0 = np.zeros((dim, dim), dtype=np.float64)
    a_1 = np.zeros((dim, dim), dtype=np.float64)

    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        h = _scaled_step(x_eq[j], eps)
        dx[j] = h
        f_plus = full_f_fn(x_eq + dx, x_eq)
        f_minus = full_f_fn(x_eq - dx, x_eq)
        a_0[:, j] = (f_plus - f_minus) / (2.0 * h)

    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        h = _scaled_step(x_eq[j], eps)
        dx[j] = h
        f_plus = full_f_fn(x_eq, x_eq + dx)
        f_minus = full_f_fn(x_eq, x_eq - dx)
        a_1[:, j] = (f_plus - f_minus) / (2.0 * h)

    return a_0, a_1