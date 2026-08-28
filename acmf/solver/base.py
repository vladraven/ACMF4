from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SolverDomain:
    """Определение границ фазового пространства для оператора отражения."""
    sid_buf: float
    sid_max: float
    f_max: float


@dataclass
class DiagnosticStep:
    """Диагностические метрики одиночного шага интегрирования."""
    lower_local_time: np.ndarray
    upper_local_time: np.ndarray
    raw_overshoot: np.ndarray
    reflected: bool
    # ИСПРАВЛЕНО: раньше клемп по Inst/Ch/Prod/M/F/Scar не диагностировался
    # вообще (в отличие от SID) — нарушение TEST_01 ("число отражений...
    # для всех bounded-переменных") и молчаливое "Hardcoded clamp
    # запрещён" (§17 документа). Теперь фиксируется явно.
    ode_clamp_overshoot: np.ndarray
    ode_clamped: bool


class StochasticStepScheme(ABC):
    """Абстрактный базовый класс для схем стохастического шага."""

    @abstractmethod
    def step(
        self,
        current_state: np.ndarray,
        drift: np.ndarray,
        diffusion_sigma: np.ndarray,
        diffusion_derivative: np.ndarray,
        random_normal: np.ndarray,
        jump_vector: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Вычисляет неотраженное кандидатное состояние Y_{n+1}."""
        pass