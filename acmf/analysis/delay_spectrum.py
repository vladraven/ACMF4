from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class DelaySpectrumConfig:
    """Параметры спектрального окна поиска комплексных корней DDE."""
    lambda_min: float = -5.0
    lambda_max: float = 2.0
    omega_max: float = 10.0
    grid_re_points: int = 15
    grid_im_points: int = 21
    tolerance: float = 1e-5


@dataclass(frozen=True)
class DelaySpectrumResult:
    """Найденные комплексные корни характеристического оператора DDE."""
    roots: np.ndarray
    critical_root: complex
    is_stable: bool


class DelaySpectrumSolver:
    """
    Численный решатель трансцендентного характеристического уравнения:
    det(D(lambda)) = det(lambda*I - A_0 - A_1*exp(-lambda*Delta_t)) = 0
    """

    def __init__(self, config: DelaySpectrumConfig | None = None) -> None:
        self.config = config or DelaySpectrumConfig()

    def _char_operator_norm(
        self,
        z: np.ndarray,
        a_0: np.ndarray,
        a_1: np.ndarray,
        delay: float,
    ) -> float:
        lam = complex(z[0], z[1])
        dim = a_0.shape[0]
        i_mat = np.eye(dim, dtype=np.complex128)
        d_lam = lam * i_mat - a_0 - a_1 * np.exp(-lam * delay)
        # Ищем корни через минимальное сингулярное число (det(D) == 0 <=> min(svd) == 0)
        s = np.linalg.svd(d_lam, compute_uv=False)
        return float(s[-1])

    def find_roots(
        self,
        a_0: np.ndarray,
        a_1: np.ndarray,
        delay: float,
    ) -> DelaySpectrumResult:
        """Сканирует спектральное окно и находит уникальные корни характеристического оператора."""
        cfg = self.config
        re_grid = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.grid_re_points)
        im_grid = np.linspace(-cfg.omega_max, cfg.omega_max, cfg.grid_im_points)

        found_roots: list[complex] = []

        for re_val in re_grid:
            for im_val in im_grid:
                res = minimize(
                    self._char_operator_norm,
                    x0=np.array([re_val, im_val]),
                    args=(a_0, a_1, delay),
                    method="Nelder-Mead",
                    tol=1e-7,
                )
                if res.fun < cfg.tolerance:
                    root = complex(round(res.x[0], 4), round(res.x[1], 4))
                    # Проверка на дубликаты
                    if not any(abs(root - r) < 1e-3 for r in found_roots):
                        found_roots.append(root)

        if not found_roots:
            # Резервный расчет мгновенного корня
            fallback = np.linalg.eigvals(a_0 + a_1)
            found_roots = [complex(x) for x in fallback]

        roots_arr = np.array(found_roots, dtype=np.complex128)
        crit_idx = int(np.argmax(np.real(roots_arr)))
        crit_root = roots_arr[crit_idx]
        is_stable = bool(np.real(crit_root) < 0.0)

        return DelaySpectrumResult(
            roots=roots_arr,
            critical_root=crit_root,
            is_stable=is_stable,
        )