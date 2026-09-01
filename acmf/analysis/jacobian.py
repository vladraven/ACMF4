from typing import Callable
import numpy as np


def _scaled_step(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Возвращает вектор шагов конечных разностей, масштабированных по
    величине каждой компоненты: eps_j = eps * max(1, |x_j|).

    Фиксированный абсолютный шаг корректен только когда все компоненты
    состояния ~O(1). Компоненты ACMF (SID, F и т.д.) имеют разные
    характерные масштабы, поэтому единый абсолютный eps даёт разную
    обусловленность производной по разным координатам: либо избыточную
    ошибку усечения (шаг слишком мал относительно |x_j|), либо ошибку
    сокращения (шаг слишком велик). Относительный шаг выравнивает это.
    """
    return eps * np.maximum(1.0, np.abs(x))


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

    steps = _scaled_step(x, eps)

    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = steps[j]
        f_plus = f(x + dx)
        f_minus = f(x - dx)
        jac[:, j] = (f_plus - f_minus) / (2.0 * steps[j])

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

    Обе якобиевы матрицы линеаризуются в одной и той же точке
    равновесия x_eq (текущее и запаздывающее состояние совпадают на
    равновесии), поэтому используется общий вектор масштабированных
    шагов steps.
    """
    dim = len(x_eq)
    a_0 = np.zeros((dim, dim), dtype=np.float64)
    a_1 = np.zeros((dim, dim), dtype=np.float64)

    steps = _scaled_step(x_eq, eps)

    # Вычисление A_0 — производная по текущему состоянию x(t).
    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = steps[j]
        f_plus = full_f_fn(x_eq + dx, x_eq)
        f_minus = full_f_fn(x_eq - dx, x_eq)
        a_0[:, j] = (f_plus - f_minus) / (2.0 * steps[j])

    # Вычисление A_1 — производная по запаздывающему состоянию x(t-Delta_t).
    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = steps[j]
        f_plus = full_f_fn(x_eq, x_eq + dx)
        f_minus = full_f_fn(x_eq, x_eq - dx)
        a_1[:, j] = (f_plus - f_minus) / (2.0 * steps[j])

    return a_0, a_1
