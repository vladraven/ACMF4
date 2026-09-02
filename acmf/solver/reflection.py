import numpy as np
from acmf.solver.base import SolverDomain, DiagnosticStep


class SkorokhodReflector:
    """
    Дискретный оператор отражения Скорохода R_Omega.
    Для всех bounded-переменных явно вычисляет lower/upper local time.
    Не использует hardcoded clamp.
    """

    def __init__(self, domain: SolverDomain) -> None:
        self.domain = domain

    def reflect_state(self, raw_state: np.ndarray) -> tuple[np.ndarray, DiagnosticStep]:
        dim = len(raw_state)
        reflected = raw_state.copy()

        # --- SID [0:3] : [-SID_buf, SID_max] ---
        sid_raw = raw_state[0:3].copy()
        l_lower_sid = np.maximum(0.0, -self.domain.sid_buf - sid_raw)
        l_upper_sid = np.maximum(0.0, sid_raw - self.domain.sid_max)
        reflected_sid = sid_raw + l_lower_sid - l_upper_sid

        # --- Inst [3] : [0, 1] ---
        inst_raw = raw_state[3]
        l_lower_inst = max(0.0, -inst_raw)
        l_upper_inst = max(0.0, inst_raw - 1.0)
        reflected_inst = inst_raw + l_lower_inst - l_upper_inst

        # --- Ch [4] : [0, 1] ---
        ch_raw = raw_state[4]
        l_lower_ch = max(0.0, -ch_raw)
        l_upper_ch = max(0.0, ch_raw - 1.0)
        reflected_ch = ch_raw + l_lower_ch - l_upper_ch

        # --- Prod [5] : [0, 1] ---
        prod_raw = raw_state[5]
        l_lower_prod = max(0.0, -prod_raw)
        l_upper_prod = max(0.0, prod_raw - 1.0)
        reflected_prod = prod_raw + l_lower_prod - l_upper_prod

        # --- M [6] : [0, 1] ---
        m_raw = raw_state[6]
        l_lower_m = max(0.0, -m_raw)
        l_upper_m = max(0.0, m_raw - 1.0)
        reflected_m = m_raw + l_lower_m - l_upper_m

        # --- F [7] : [0, F_max] ---
        f_raw = raw_state[7]
        l_lower_f = max(0.0, -f_raw)
        l_upper_f = max(0.0, f_raw - self.domain.f_max)
        reflected_f = f_raw + l_lower_f - l_upper_f

        # --- Scar [8] : [0, 1] ---
        scar_raw = raw_state[8]
        l_lower_scar = max(0.0, -scar_raw)
        l_upper_scar = max(0.0, scar_raw - 1.0)
        reflected_scar = scar_raw + l_lower_scar - l_upper_scar

        # --- ED [9:12] : [0, +inf) ---
        ed_raw = raw_state[9:12].copy()
        l_lower_ed = np.maximum(0.0, -ed_raw)
        reflected_ed = ed_raw + l_lower_ed

        # --- RecDebt [12] : [0, +inf) ---
        rec_raw = raw_state[12]
        l_lower_rec = max(0.0, -rec_raw)
        reflected_rec = rec_raw + l_lower_rec

        # Сборка
        reflected[0:3] = reflected_sid
        reflected[3] = reflected_inst
        reflected[4] = reflected_ch
        reflected[5] = reflected_prod
        reflected[6] = reflected_m
        reflected[7] = reflected_f
        reflected[8] = reflected_scar
        reflected[9:12] = reflected_ed
        reflected[12] = reflected_rec

        diag = DiagnosticStep(
            lower_local_time=l_lower_sid,
            upper_local_time=l_upper_sid,
            raw_overshoot=np.array([
                max(0.0, self.domain.sid_max - sid_raw[k]) for k in range(3)
            ]),
            reflected=True,
            ode_clamp_overshoot=np.array([
                l_lower_inst, l_upper_inst,
                l_lower_ch, l_upper_ch,
                l_lower_prod, l_upper_prod,
                l_lower_m, l_upper_m,
                l_lower_f, l_upper_f,
                l_lower_scar, l_upper_scar,
            ]),
            ode_clamped=True,
        )

        return reflected, diag