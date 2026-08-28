from dataclasses import dataclass
from typing import Callable
import numpy as np
from acmf.analysis.equilibria import EquilibriumEngine, EquilibriumPoint
from acmf.analysis.jacobian import compute_dde_jacobians
from acmf.analysis.delay_spectrum import DelaySpectrumSolver


@dataclass(frozen=True)
class BifurcationPoint:
    """Обнаруженная точка бифуркации при сканировании параметра."""
    bifurcation_type: str  # 'Saddle-Node', 'Hopf', или 'None'
    parameter_value: float
    equilibrium_state: np.ndarray
    critical_eigenvalue: complex
    transversality: float


class ContinuationEngine:
    """
    Параметрическое сканирование и обнаружение бифуркаций (Saddle-Node, Hopf).
    """

    def __init__(
        self,
        eq_engine: EquilibriumEngine | None = None,
        delay_solver: DelaySpectrumSolver | None = None,
    ) -> None:
        self.eq_engine = eq_engine or EquilibriumEngine()
        self.delay_solver = delay_solver or DelaySpectrumSolver()

    def scan_parameter(
        self,
        drift_factory: Callable[[float], tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]],
        param_values: np.ndarray,
        initial_state: np.ndarray,
        delay: float = 0.0,
    ) -> list[BifurcationPoint]:
        """Сканирует параметр и фиксирует спектральные пересечения мнимой оси."""
        bifurcations: list[BifurcationPoint] = []
        curr_guess = initial_state.copy()

        prev_re = None
        prev_param = None

        for p_val in param_values:
            drift_det, drift_full = drift_factory(p_val)
            eq = self.eq_engine.find_equilibrium(drift_det, curr_guess)

            if not eq.is_valid:
                continue

            curr_guess = eq.state.copy()
            a_0, a_1 = compute_dde_jacobians(drift_full, eq.state)
            spec = self.delay_solver.find_roots(a_0, a_1, delay)
            crit_root = spec.critical_root
            curr_re = np.real(crit_root)

            if prev_re is not None and prev_param is not None:
                # Проверка пересечения мнимой оси (смена знака Re(lambda))
                if (prev_re < 0.0 <= curr_re) or (prev_re > 0.0 >= curr_re):
                    dp = p_val - prev_param
                    transversality = (curr_re - prev_re) / dp if dp != 0.0 else 0.0
                    bif_type = "Hopf" if abs(np.imag(crit_root)) > 1e-2 else "Saddle-Node"

                    bifurcations.append(
                        BifurcationPoint(
                            bifurcation_type=bif_type,
                            parameter_value=float(p_val),
                            equilibrium_state=eq.state,
                            critical_eigenvalue=crit_root,
                            transversality=float(transversality),
                        )
                    )

            prev_re = curr_re
            prev_param = p_val

        return bifurcations