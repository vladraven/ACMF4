from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InstantaneousSpectrumResult:
    """Результаты спектрального анализа матрицы Якоби J_inst."""
    eigenvalues: np.ndarray
    critical_eigenvalue: complex
    right_critical_vector: np.ndarray
    left_critical_vector: np.ndarray
    is_stable: bool


def analyze_instantaneous_spectrum(jacobian: np.ndarray) -> InstantaneousSpectrumResult:
    """
    Вычисляет мгновенный спектр Якобиана, ведущую моду и биортогональные векторы:
    w^H * v = 1.
    """
    vals, v_mat = np.linalg.eig(jacobian)
    w_mat = np.linalg.inv(v_mat).T.conj()

    max_idx = int(np.argmax(np.real(vals)))
    crit_val = vals[max_idx]

    v_crit = v_mat[:, max_idx]
    w_crit = w_mat[:, max_idx]

    # Биортогональная нормировка w^H * v = 1
    dot_product = np.vdot(w_crit, v_crit)
    if abs(dot_product) > 1e-12:
        w_crit = w_crit / dot_product.conj()

    is_stable = bool(np.real(crit_val) < 0.0)

    return InstantaneousSpectrumResult(
        eigenvalues=vals,
        critical_eigenvalue=complex(crit_val),
        right_critical_vector=v_crit,
        left_critical_vector=w_crit,
        is_stable=is_stable,
    )