import numpy as np
from acmf.solver.base import SolverDomain, DiagnosticStep


class SkorokhodReflector:
    """
    Численный оператор отражения Скорохода R_Omega.
    Компенсирует стохастический overshoot по нормали к границам домена:
    X_{n+1} = Y_{n+1} + L_lower - L_upper
    """

    def __init__(self, domain: SolverDomain) -> None:
        self.domain = domain

    def reflect_sid(self, raw_sid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Применяет отражение к компонентам SID: [-SID_buf, SID_max].
        Возвращает (reflected_sid, L_lower, L_upper).
        """
        l_lower = np.maximum(0.0, -self.domain.sid_buf - raw_sid)
        l_upper = np.maximum(0.0, raw_sid - self.domain.sid_max)
        reflected_sid = raw_sid + l_lower - l_upper
        return reflected_sid, l_lower, l_upper

    def reflect_state(self, raw_state: np.ndarray) -> tuple[np.ndarray, DiagnosticStep]:
        """
        Отражает полное 13-мерное состояние в допустимый домен Omega.
        """
        reflected = raw_state.copy()

        # 1. SID компоненты [0:3]
        reflected_sid, l_lower, l_upper = self.reflect_sid(raw_state[0:3])
        reflected[0:3] = reflected_sid

        # 2. Ограниченные переменные [Inst, Ch, Prod, M, Scar] в [0, 1]
        for idx in [3, 4, 5, 6, 8]:
            if reflected[idx] < 0.0:
                reflected[idx] = 0.0
            elif reflected[idx] > 1.0:
                reflected[idx] = 1.0

        # 3. Физический капитал F в [0, F_max]
        if reflected[7] < 0.0:
            reflected[7] = 0.0
        elif reflected[7] > self.domain.f_max:
            reflected[7] = self.domain.f_max

        # 4. Неотрицательные переменные [ED^1, ED^2, ED^3, RecDebt]
        reflected[9:12] = np.maximum(0.0, reflected[9:12])
        reflected[12] = max(0.0, float(reflected[12]))

        overshoot = np.abs(raw_state[0:3] - reflected[0:3])
        has_reflected = bool(np.any(l_lower > 0.0) or np.any(l_upper > 0.0))

        diag = DiagnosticStep(
            lower_local_time=l_lower,
            upper_local_time=l_upper,
            raw_overshoot=overshoot,
            reflected=has_reflected,
        )

        return reflected, diag