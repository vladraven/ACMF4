from dataclasses import dataclass
from typing import Callable
import math


@dataclass(frozen=True)
class ForcingState:
    """Значения экзогенных воздействий в текущий момент времени t."""
    A: float
    R_0: float
    V: float
    G: float
    dA_dt: float = 0.0

    def validate(self) -> None:
        values = (self.A, self.R_0, self.V, self.G, self.dA_dt)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Forcing содержит NaN или бесконечность")
        if self.A < 0.0:
            raise ValueError(f"Forcing A должно быть >= 0, получено {self.A}")
        if self.R_0 < 0.0:
            raise ValueError(
                f"Forcing R_0 должно быть >= 0, получено {self.R_0}"
            )
        if not (0.0 <= self.V <= 1.0):
            raise ValueError(
                f"Forcing V должно быть в [0, 1], получено {self.V}"
            )
        if not (0.0 <= self.G <= 1.0):
            raise ValueError(
                f"Forcing G должно быть в [0, 1], получено {self.G}"
            )


class ForcingProfile:
    """Временной профиль экзогенных функций."""

    def __init__(
        self,
        A_fn: Callable[[float], float] | None = None,
        R0_fn: Callable[[float], float] | None = None,
        V_fn: Callable[[float], float] | None = None,
        G_fn: Callable[[float], float] | None = None,
        dA_dt_fn: Callable[[float], float] | None = None,
    ) -> None:
        self._A_fn = A_fn or (lambda t: 1.0)
        self._R0_fn = R0_fn or (lambda t: 1.0)
        self._V_fn = V_fn or (lambda t: 1.0)
        self._G_fn = G_fn or (lambda t: 1.0)
        self._dA_dt_fn = dA_dt_fn or (lambda t: 0.0)

    @property
    def dA_dt_fn(self) -> Callable[[float], float]:
        return self._dA_dt_fn

    def evaluate(self, t: float) -> ForcingState:
        state = ForcingState(
            A=float(self._A_fn(t)),
            R_0=float(self._R0_fn(t)),
            V=float(self._V_fn(t)),
            G=float(self._G_fn(t)),
            dA_dt=float(self._dA_dt_fn(t)),
        )
        state.validate()
        return state
