import numpy as np


def compute_rolling_variance(series: np.ndarray, window_size: int) -> np.ndarray:
    """Вычисляет скользящую дисперсию Var(Z) временного ряда."""
    n = len(series)
    if n < window_size or window_size < 2:
        return np.zeros(n, dtype=np.float64)

    var_series = np.zeros(n, dtype=np.float64)
    for i in range(window_size, n + 1):
        window = series[i - window_size : i]
        var_series[i - 1] = float(np.var(window, ddof=1))

    var_series[: window_size - 1] = var_series[window_size - 1]
    return var_series