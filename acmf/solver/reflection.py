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

        # 1. SID компоненты [0:3] — истинное отражение Скорохода
        reflected_sid, l_lower, l_upper = self.reflect_sid(raw_state[0:3])
        reflected[0:3] = reflected_sid

        # 2-3. Ограниченные ОДУ-переменные [Inst, Ch, Prod, M, F, Scar].
        # Эти переменные по построению (см. drift.py/dynamics.py — inward
        # -pointing drift на обеих границах, доказано аналитически) не
        # должны покидать домен при корректном интегрировании: на них не
        # действуют ни диффузия, ни скачки. Клемп здесь — не смоделированный
        # механизм, а численный fallback на случай overshoot от конечного
        # dt. Раньше он был немым; теперь величина overshoot фиксируется
        # в диагностике для всех шести переменных, как того явно требует
        # TEST_01 документа. Ненулевое значение здесь на реальном прогоне
        # означает, что dt слишком велик относительно alpha/beta/mu, либо
        # есть ошибка знака в одном из drift-термов — и не должно
        # игнорироваться молча.
        ode_indices = [3, 4, 5, 6, 7, 8]  # Inst, Ch, Prod, M, F, Scar
        ode_upper_bounds = {7: self.domain.f_max}  # остальные — [0, 1]
        ode_overshoot = np.zeros(len(ode_indices), dtype=np.float64)
        for pos, idx in enumerate(ode_indices):
            upper = ode_upper_bounds.get(idx, 1.0)
            raw_val = reflected[idx]
            if raw_val < 0.0:
                ode_overshoot[pos] = -raw_val
                reflected[idx] = 0.0
            elif raw_val > upper:
                ode_overshoot[pos] = raw_val - upper
                reflected[idx] = upper

        # 4. Неотрицательные переменные [ED^1, ED^2, ED^3, RecDebt]
        reflected[9:12] = np.maximum(0.0, reflected[9:12])
        reflected[12] = max(0.0, float(reflected[12]))

        overshoot = np.abs(raw_state[0:3] - reflected[0:3])
        has_reflected = bool(np.any(l_lower > 0.0) or np.any(l_upper > 0.0))
        ode_clamped = bool(np.any(ode_overshoot > 0.0))

        diag = DiagnosticStep(
            lower_local_time=l_lower,
            upper_local_time=l_upper,
            raw_overshoot=overshoot,
            reflected=has_reflected,
            ode_clamp_overshoot=ode_overshoot,
            ode_clamped=ode_clamped,
        )

        return reflected, diag