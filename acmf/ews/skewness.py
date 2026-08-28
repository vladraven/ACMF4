import numpy as np
from scipy.stats import skew


def compute_rolling_skewness(series: np.ndarray, window_size: int) -> np.ndarray:
    """Вычисляет скользящую асимметрию Skewness(Z)."""
    n = len(series)
    if n < window_size or window_size < 3:
        return np.zeros(n, dtype=np.float64)

    skew_series = np.zeros(n, dtype=np.float64)
    for i in range(window_size, n + 1):
        window = series[i - window_size : i]
        skew_series[i - 1] = float(skew(window, bias=False))

    skew_series[: window_size - 1] = skew_series[window_size - 1]
    return skew_series