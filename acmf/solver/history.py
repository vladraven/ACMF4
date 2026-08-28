import numpy as np


class HistoryBuffer:
    """Causal ring buffer for state, state derivatives and exogenous dA/dt."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        if capacity < 2:
            raise ValueError("History capacity must be >= 2")
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.time_buffer = np.zeros(capacity, dtype=np.float64)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float64)
        self.deriv_buffer = np.zeros((capacity, state_dim), dtype=np.float64)
        self.d_a_dt_buffer = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.cursor = 0

    def append(self, t: float, state: np.ndarray, deriv: np.ndarray, d_a_dt: float = 0.0) -> None:
        state = np.asarray(state, dtype=np.float64)
        deriv = np.asarray(deriv, dtype=np.float64)
        if state.shape != (self.state_dim,) or deriv.shape != (self.state_dim,):
            raise ValueError(f"Expected state/deriv shape {(self.state_dim,)}, got {state.shape}/{deriv.shape}")
        self.time_buffer[self.cursor] = float(t)
        self.state_buffer[self.cursor] = state
        self.deriv_buffer[self.cursor] = deriv
        self.d_a_dt_buffer[self.cursor] = float(d_a_dt)
        self.cursor = (self.cursor + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _ordered_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return np.roll(np.arange(self.capacity), -self.cursor)

    def get_ordered_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self._ordered_indices()
        return self.time_buffer[idx], self.state_buffer[idx], self.deriv_buffer[idx]

    def get_ordered_forcing_history(self) -> tuple[np.ndarray, np.ndarray]:
        idx = self._ordered_indices()
        return self.time_buffer[idx], self.d_a_dt_buffer[idx]
