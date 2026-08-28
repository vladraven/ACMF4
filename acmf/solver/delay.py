import numpy as np
from acmf.solver.history import HistoryBuffer


class CausalDelayLookup:
    @staticmethod
    def _interpolate_scalar(times: np.ndarray, values: np.ndarray, target_t: float) -> float:
        if len(times) == 0:
            raise RuntimeError("History buffer is empty")
        if target_t <= times[0]:
            return float(values[0])
        if target_t >= times[-1]:
            return float(values[-1])
        idx = int(np.searchsorted(times, target_t))
        t_prev, t_next = times[idx - 1], times[idx]
        if t_next == t_prev:
            return float(values[idx])
        weight = (target_t - t_prev) / (t_next - t_prev)
        return float((1.0 - weight) * values[idx - 1] + weight * values[idx])

    @staticmethod
    def get_delayed_state(history: HistoryBuffer, current_time: float, delay: float) -> np.ndarray:
        target_t = current_time - delay
        times, states, _ = history.get_ordered_history()
        if len(times) == 0:
            raise RuntimeError("History buffer is empty")
        if target_t <= times[0]: return states[0].copy()
        if target_t >= times[-1]: return states[-1].copy()
        idx = int(np.searchsorted(times, target_t))
        t_prev, t_next = times[idx - 1], times[idx]
        if t_next == t_prev: return states[idx].copy()
        weight = (target_t - t_prev) / (t_next - t_prev)
        return (1.0 - weight) * states[idx - 1] + weight * states[idx]

    @staticmethod
    def get_delayed_derivative(history: HistoryBuffer, current_time: float, delay: float, component_idx: int) -> float:
        times, _, derivs = history.get_ordered_history()
        return CausalDelayLookup._interpolate_scalar(times, derivs[:, component_idx], current_time - delay)

    @staticmethod
    def get_delayed_forcing_derivative(history: HistoryBuffer, current_time: float, delay: float) -> float:
        times, d_a_dt = history.get_ordered_forcing_history()
        return CausalDelayLookup._interpolate_scalar(times, d_a_dt, current_time - delay)
