import numpy as np
from acmf.solver.history import HistoryBuffer


class CausalDelayLookup:
    """
    Модуль каузального извлечения запаздывающих значений из буфера истории
    с использованием линейной интерполяции.
    """

    @staticmethod
    def get_delayed_state(
        history: HistoryBuffer,
        current_time: float,
        delay: float,
    ) -> np.ndarray:
        """Извлекает состояние X(t - delay)."""
        target_t = current_time - delay
        times, states, _ = history.get_ordered_history()

        if len(times) == 0:
            raise RuntimeError("Буфер истории пуст")

        if target_t <= times[0]:
            return states[0].copy()
        if target_t >= times[-1]:
            return states[-1].copy()

        idx = np.searchsorted(times, target_t)
        t_prev, t_next = times[idx - 1], times[idx]
        weight = (target_t - t_prev) / (t_next - t_prev)

        return (1.0 - weight) * states[idx - 1] + weight * states[idx]

    @staticmethod
    def get_delayed_derivative(
        history: HistoryBuffer,
        current_time: float,
        delay: float,
        component_idx: int,
    ) -> float:
        """Извлекает производную dX_k/dt(t - delay)."""
        target_t = current_time - delay
        times, _, derivs = history.get_ordered_history()

        if len(times) == 0:
            raise RuntimeError("Буфер истории пуст")

        if target_t <= times[0]:
            return float(derivs[0, component_idx])
        if target_t >= times[-1]:
            return float(derivs[-1, component_idx])

        idx = np.searchsorted(times, target_t)
        t_prev, t_next = times[idx - 1], times[idx]
        weight = (target_t - t_prev) / (t_next - t_prev)

        val = (1.0 - weight) * derivs[idx - 1, component_idx] + weight * derivs[idx, component_idx]
        return float(val)