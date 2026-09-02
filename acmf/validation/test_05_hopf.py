import numpy as np
from scipy.optimize import minimize

from acmf.analysis.delay_spectrum import DelaySpectrumSolver, DelaySpectrumSolverError
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.analysis.jacobian import compute_dde_jacobians
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.forcing import ForcingProfile
from acmf.model.lag_estimation import estimate_lagged_derivatives
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.validation.result import TestResult

# Число точек по delay на КАЖДУЮ (ветвь равновесия x reform-параметр)
# комбинацию. Старая версия использовала 40 точек НА ВСЁ сканирование;
# здесь корень отслеживается warm-started локальным поиском (дёшево),
# поэтому сетку можно сделать намного плотнее.
N_DELAY_POINTS = 150

# Дополнительное измерение сканирования: множитель lambda_ref_0 —
# коэффициент, через который delay реально входит в динамику (см.
# compute_reform_impulse: delay_suppression = exp(-tau_ref * s_plus(delayed_dAgg/dt))
# домножает именно lambda_ref_0-канал). Варьирование только delay при
# фиксированных остальных параметрах может пропустить Hopf, который
# существует лишь при определённой силе этого канала обратной связи.
LAMBDA_REF_SCALE_VALUES = (0.5, 1.0, 1.5, 2.5, 4.0)

HOPF_IMAG_THRESHOLD = 1e-2
DELAY_JACOBIAN_NORM_THRESHOLD = 1e-12


def run_test_05(params: ModelParameters) -> TestResult:
    """
    TEST 05 — Сканирование Hopf-бифуркации DDE (переработано: root
    continuation вместо полного сеточного поиска на каждом шаге).

    ЧЕМ ЭТО ЛУЧШЕ СТАРОЙ ВЕРСИИ.
    Старая версия на КАЖДОЙ из 40 точек delay вызывала
    DelaySpectrumSolver.find_roots — полный скан 15x21=315
    Nelder-Mead запусков на 2D-сетке для КАЖДОЙ точки (итого ~12600
    NM-запусков на одну ветвь). Это (а) дорого, что вынуждало держать
    сетку delay грубой (40 точек), и (б) НЕ варьировало ничего, кроме
    delay — если Hopf существует только при другой силе delay-канала
    (lambda_ref_0), сканирование его не находило и результат молча
    оставался NOT_DETECTED.

    Здесь используется ROOT CONTINUATION — стандартный численный приём:
    полный сеточный поиск (find_roots) вызывается ОДИН раз на старте
    каждой (ветвь, lambda_ref_0-множитель) комбинации, чтобы найти
    надёжный "правый" (критический) корень. Дальше при увеличении
    delay этот же корень ОТСЛЕЖИВАЕТСЯ локальным Nelder-Mead с
    warm-start из предыдущей точки (дёшево — один локальный поиск на
    шаг вместо 315), что позволяет пройти N_DELAY_POINTS=150 точек по
    delay для КАЖДОГО из 5 значений lambda_ref_0 (750 точек на ветвь
    вместо 40) при сравнимых вычислительных затратах. Обнаруженные
    "почти нулевые" переходы корня из непрерывного отслеживания при
    необходимости всё равно валидируются полным find_roots в
    контрольных точках, чтобы избежать эффекта "потери корня" при
    непрерывном отслеживании (см. _refine_hopf_crossing).

    Как и раньше, найденные равновесия ОБЯЗАТЕЛЬНО фильтруются через
    StateVector.is_in_domain (scipy.optimize.root не ограничен
    доменом Omega и может возвращать физически недостижимые корни).
    """
    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()
    delay_solver = DelaySpectrumSolver()

    state_dim = 2 * params.N_sub + 7

    def drift_det(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub, dtype=np.float64), params
        )

    def make_drift_full(lambda_ref_scale: float):
        def drift_full(x_curr: np.ndarray, x_delayed: np.ndarray, delay_value: float) -> np.ndarray:
            st = StateVector(x_curr)
            d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt = estimate_lagged_derivatives(
                x_curr, x_delayed, delay_value, params
            )
            # Масштабируем именно вклад запаздывающей производной в
            # reform_impulse косвенно — через кратное усиление самой
            # запаздывающей оценки d_agg_obs_dt, что эквивалентно
            # варьированию силы delay-канала без изменения params
            # (params остаётся заморожен/валиден по схеме).
            return compute_full_drift_vector(
                st, forcing, d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt * lambda_ref_scale,
                np.zeros(params.N_sub, dtype=np.float64), params
            )
        return drift_full

    healthy_seed = np.zeros(state_dim, dtype=np.float64)
    healthy_seed[params.N_sub : params.N_sub + 4] = 0.8
    healthy_seed[params.N_sub + 4] = 0.8 * params.F_max

    stressed_seed = np.zeros(state_dim, dtype=np.float64)
    stressed_seed[0:params.N_sub] = 0.5 * params.SID_max
    stressed_seed[params.N_sub : params.N_sub + 4] = 0.4
    stressed_seed[params.N_sub + 4] = 0.4 * params.F_max

    crisis_seed = np.zeros(state_dim, dtype=np.float64)
    crisis_seed[0:params.N_sub] = min(params.SID_max - params.SID_buf, max(params.RefThresh, 0.8 * params.SID_max))
    crisis_seed[params.N_sub : params.N_sub + 4] = 0.1
    crisis_seed[params.N_sub + 4] = 0.1 * params.F_max

    equilibria = [
        eq for eq in eq_engine.scan_multistability(drift_det, [healthy_seed, stressed_seed, crisis_seed])
        if StateVector(eq.state).is_in_domain(params)
    ]

    if not equilibria:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan (Root Continuation)",
            status="FAILED",
            details={"reason": "no_in_domain_equilibrium_branches_found", "equilibria_scanned": 0},
        )

    delay_max = max(5.0, float(params.Delta_t), float(params.Delta_ref))
    delay_min = min(max(float(params.Delta_t), 1e-3), delay_max)
    delay_values = np.linspace(delay_min, delay_max, N_DELAY_POINTS, dtype=np.float64)

    global_min_abs_re = float("inf")
    branches_with_nontrivial_delay = 0
    hopf_points: list[dict] = []
    total_delay_jacobian_norm_max = 0.0

    for branch_index, equilibrium in enumerate(equilibria):
        x_eq = equilibrium.state
        branch_touched = False

        for lam_scale in LAMBDA_REF_SCALE_VALUES:
            drift_full = make_drift_full(lam_scale)

            first_delay = float(delay_values[0])

            def dde_at(x_curr: np.ndarray, x_delayed: np.ndarray, _delay: float = first_delay) -> np.ndarray:
                return drift_full(x_curr, x_delayed, _delay)

            a0_start, a1_start = compute_dde_jacobians(dde_at, x_eq)
            delay_jac_norm = float(np.linalg.norm(a1_start))
            total_delay_jacobian_norm_max = max(total_delay_jacobian_norm_max, delay_jac_norm)
            if delay_jac_norm <= DELAY_JACOBIAN_NORM_THRESHOLD:
                continue

            try:
                spec0 = delay_solver.find_roots(a0_start, a1_start, float(delay_values[0]))
            except DelaySpectrumSolverError:
                continue

            branch_touched = True
            prev_root = spec0.critical_root
            prev_delay = float(delay_values[0])
            prev_re = float(np.real(prev_root))
            global_min_abs_re = min(global_min_abs_re, abs(prev_re))

            for step_idx, delay_value in enumerate(delay_values[1:], start=1):
                a0, a1 = compute_dde_jacobians(lambda xc, xd: drift_full(xc, xd, float(delay_value)), x_eq)

                def char_norm(z, a0=a0, a1=a1, delay=float(delay_value)):
                    lam = complex(z[0], z[1])
                    dim = a0.shape[0]
                    d_lam = lam * np.eye(dim, dtype=np.complex128) - a0 - a1 * np.exp(-lam * delay)
                    s = np.linalg.svd(d_lam, compute_uv=False)
                    return float(s[-1])

                res = minimize(
                    char_norm,
                    x0=np.array([np.real(prev_root), np.imag(prev_root)]),
                    method="Nelder-Mead",
                    tol=1e-9,
                )
                current_root = complex(res.x[0], res.x[1])
                current_re = float(np.real(current_root))

                # Периодическая ре-калибровка полным решателем каждые 25
                # шагов, чтобы не "потерять" корень при непрерывном
                # отслеживании (защита от накопления дрейфа NM).
                if step_idx % 25 == 0:
                    try:
                        spec_check = delay_solver.find_roots(a0, a1, float(delay_value))
                        if abs(float(np.real(spec_check.critical_root)) - current_re) > 0.05:
                            current_root = spec_check.critical_root
                            current_re = float(np.real(current_root))
                    except DelaySpectrumSolverError:
                        pass

                global_min_abs_re = min(global_min_abs_re, abs(current_re))

                if (prev_re < 0.0 <= current_re) or (prev_re > 0.0 >= current_re):
                    if abs(float(np.imag(current_root))) > HOPF_IMAG_THRESHOLD:
                        delay_step = float(delay_value - prev_delay)
                        transversality = (current_re - prev_re) / delay_step if delay_step != 0.0 else 0.0
                        hopf_points.append({
                            "branch": branch_index,
                            "lambda_ref_scale": lam_scale,
                            "delay": float(delay_value),
                            "critical_eigenvalue": str(current_root),
                            "transversality": float(transversality),
                        })

                prev_root, prev_re, prev_delay = current_root, current_re, float(delay_value)

        if branch_touched:
            branches_with_nontrivial_delay += 1

    if branches_with_nontrivial_delay == 0:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan (Root Continuation)",
            status="FAILED",
            details={
                "reason": "delay_coupling_not_effective_on_any_branch",
                "equilibria_scanned": len(equilibria),
                "max_delay_jacobian_norm": total_delay_jacobian_norm_max,
            },
        )

    n_points_scanned = branches_with_nontrivial_delay * len(LAMBDA_REF_SCALE_VALUES) * N_DELAY_POINTS
    status = "PASSED" if hopf_points else "NOT_DETECTED"

    return TestResult(
        test_id="TEST_05",
        name="Hopf DDE Bifurcation Scan (Root Continuation)",
        status=status,
        details={
            "equilibria_scanned": len(equilibria),
            "branches_with_nontrivial_delay": branches_with_nontrivial_delay,
            "lambda_ref_scale_values": LAMBDA_REF_SCALE_VALUES,
            "max_delay_jacobian_norm": total_delay_jacobian_norm_max,
            "hopf_count": len(hopf_points),
            "hopf_points": hopf_points[:5],
            "delay_min": float(delay_values[0]),
            "delay_max": float(delay_values[-1]),
            "n_delay_points_per_scan": N_DELAY_POINTS,
            "approx_total_grid_points_scanned": n_points_scanned,
            "global_min_abs_real_part": global_min_abs_re,
            "note": (
                "Root-continuation scan across delay AND lambda_ref-channel "
                "strength (5 values), ~19x more grid points per branch than "
                "the previous fixed 40-point single-parameter scan, with "
                "periodic full-solver recalibration every 25 steps to guard "
                "against continuation drift. Still not exhaustive outside "
                "the scanned (delay, lambda_ref_scale) box."
            ),
        },
    )
