import numpy as np


def s_plus(x: np.ndarray | float, kappa_s: float) -> np.ndarray | float:
    """Численно устойчивый гладкий оператор S+(x; kappa_s)."""
    return np.logaddexp(0.0, kappa_s * x) / kappa_s


def s_minus(x: np.ndarray | float, kappa_s: float) -> np.ndarray | float:
    """Численно устойчивый гладкий оператор S-(x; kappa_s)."""
    return -s_plus(-x, kappa_s)