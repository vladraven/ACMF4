import numpy as np


class HistoryBuffer:
    """
    Кольцевой буфер истории траектории фиксированной длины для каузального DDE-доступа.
    Хранит временные метки, векторы состояния X(t) и производные dX/dt.
    """

    def __init__(self, capacity: int, state_dim: int = 13) -> None:
        if capacity < 2:
            raise ValueError("Вместимость буфера истории должна быть >= 2")
        self.capacity = capacity
        self.state_dim = state_dim

        self.time_buffer = np.zeros(capacity, dtype=np.float64)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float64)
        self.deriv_buffer = np.zeros((capacity, state_dim), dtype=np.float64)

        self.size = 0
        self.cursor = 0

    def append(self, t: float, state: np.ndarray, deriv: np.ndarray) -> None:
        """Добавляет новый временной срез в кольцевой буфер."""
        self.time_buffer[self.cursor] = t
        self.state_buffer[self.cursor] = state
        self.deriv_buffer[self.cursor] = deriv

        self.cursor = (self.cursor + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_ordered_history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает историю, отсортированную в хронологическом порядке."""
        if self.size < self.capacity:
            return (
                self.time_buffer[: self.size],
                self.state_buffer[: self.size],
                self.deriv_buffer[: self.size],
            )
        indices = np.roll(np.arange(self.capacity), -self.cursor)
        return (
            self.time_buffer[indices],
            self.state_buffer[indices],
            self.deriv_buffer[indices],
        )