import numpy as np


def compute_rolling_ar1(series: np.ndarray, window_size: int) -> np.ndarray:
    """Вычисляет скользящую автокорреляцию первого порядка AR(1) ряда Z(t)."""
    n = len(series)
    if n < window_size or window_size < 3:
        return np.zeros(n, dtype=np.float64)

    ar1_series = np.zeros(n, dtype=np.float64)
    for i in range(window_size, n + 1):
        window = series[i - window_size : i]
        x_prev = window[:-1] - np.mean(window[:-1])
        x_curr = window[1:] - np.mean(window[1:])
        denom = np.sum(x_prev**2)
        if denom > 1e-12:
            ar1_series[i - 1] = float(np.sum(x_prev * x_curr) / denom)
        else:
            ar1_series[i - 1] = 0.0

    ar1_series[: window_size - 1] = ar1_series[window_size - 1]
    return ar1_series