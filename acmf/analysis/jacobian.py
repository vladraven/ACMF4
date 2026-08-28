from typing import Callable
import numpy as np


def compute_finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Вычисляет Якобиан вектор-функции F(X) центральными конечными разностями.
    Размерность выхода: (dim, dim).
    """
    dim = len(x)
    f0 = f(x)
    dim_out = len(f0)
    jac = np.zeros((dim_out, dim), dtype=np.float64)

    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = eps
        f_plus = f(x + dx)
        f_minus = f(x - dx)
        jac[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return jac


def compute_dde_jacobians(
    full_f_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_eq: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет матрицы линеаризации DDE системы в точке равновесия:
    dx/dt = A_0 * x(t) + A_1 * x(t - Delta_t)
    где A_0 = dF/dX(t), A_1 = dF/dX(t - Delta_t).
    """
    dim = len(x_eq)
    a_0 = np.zeros((dim, dim), dtype=np.float64)
    a_1 = np.zeros((dim, dim), dtype=np.float64)

    # Вычисление A_0
    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = eps
        f_plus = full_f_fn(x_eq + dx, x_eq)
        f_minus = full_f_fn(x_eq - dx, x_eq)
        a_0[:, j] = (f_plus - f_minus) / (2.0 * eps)

    # Вычисление A_1
    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = eps
        f_plus = full_f_fn(x_eq, x_eq + dx)
        f_minus = full_f_fn(x_eq, x_eq - dx)
        a_1[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return a_0, a_1