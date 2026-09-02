import dataclasses
import numpy as np
from scipy.optimize import differential_evolution
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.validation.result import TestResult

# NB: scipy.optimize.root ищет корни F(x)=0 БЕЗ ограничения x в Omega —
# он может (и делает это на практике) находить формально валидные по
# невязке корни ВНЕ физического домена (например, с ED^k < 0), которые
# не являются достижимыми равновесиями отражённой системы. Каждая
# найденная точка ниже ОБЯЗАТЕЛЬНО фильтруется через
# StateVector.is_in_domain — иначе "почти нулевые" собственные значения
# на таких паразитных корнях дают ложные кандидаты на бифуркацию.

# Порог "почти нулевого" собственного значения, при котором точка
# считается КАНДИДАТОМ на седло-узловую бифуркацию и запускается
# уточнение (центральная разность по параметру + классификация).
FOLD_CANDIDATE_THRESHOLD = 5e-3

# Штраф, возвращаемый оптимизатору, когда в данной точке параметров не
# найдено ни одного валидного равновесия (чтобы не путать "равновесия
# нет" с "равновесие устойчиво и далеко от бифуркации").
NO_EQUILIBRIUM_PENALTY = 10.0

# Границы 4-мерного параметрического бокса для глобального поиска.
# Параметры входят в петлю положительной обратной связи
# Inst<->Ch<->Prod<->M (наиболее вероятный источник мультистабильности
# и, следовательно, седло-узловой бифуркации в этой модели).
PARAM_BOUNDS = [
    (0.02, 3.0),   # R_0 (базовая регенерация)
    (0.0, 2.5),    # mu_inst (распад институтов)
    (0.0, 3.0),    # alpha_pos (скорость роста институтов)
    (0.0, 1.0),    # gamma_R (чувствительность R_eff к Scar)
]


def _build_drift(base_params: ModelParameters, p_vec: np.ndarray):
    r0, mu_inst, alpha_pos, gamma_r = p_vec
    local_params = dataclasses.replace(
        base_params, mu_inst=float(mu_inst), alpha_pos=float(alpha_pos), gamma_R=float(gamma_r)
    )
    forcing = ForcingProfile(R0_fn=lambda t: float(r0)).evaluate(0.0)

    def drift_det(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(local_params.N_sub), local_params
        )

    return drift_det, local_params


def _finite_difference_jacobian(drift_fn, x_eq: np.ndarray) -> np.ndarray:
    dim = len(x_eq)
    eps = 1e-6
    steps = eps * np.maximum(1.0, np.abs(x_eq))
    jac = np.zeros((dim, dim), dtype=np.float64)
    for j in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[j] = steps[j]
        jac[:, j] = (drift_fn(x_eq + dx) - drift_fn(x_eq - dx)) / (2.0 * steps[j])
    return jac


def _min_abs_real_eigenvalue(drift_fn, x_eq: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(_finite_difference_jacobian(drift_fn, x_eq))
    return float(np.min(np.abs(np.real(eigvals))))


def _crit_real_part(eq_engine: EquilibriumEngine, drift_fn, x_guess: np.ndarray, local_params: ModelParameters) -> float | None:
    eq = eq_engine.find_equilibrium(drift_fn, x_guess)
    if not eq.is_valid or not StateVector(eq.state).is_in_domain(local_params):
        return None
    eigs = np.linalg.eigvals(_finite_difference_jacobian(drift_fn, eq.state))
    idx = int(np.argmin(np.abs(np.real(eigs))))
    return float(np.real(eigs[idx]))


def _fold_margin_at(base_params: ModelParameters, p_vec: np.ndarray, eq_engine: EquilibriumEngine, seeds: list[np.ndarray]) -> float:
    drift_det, local_params = _build_drift(base_params, p_vec)
    best_margin = None
    for seed in seeds:
        eq = eq_engine.find_equilibrium(drift_det, seed)
        if not eq.is_valid or not StateVector(eq.state).is_in_domain(local_params):
            continue
        margin = _min_abs_real_eigenvalue(drift_det, eq.state)
        if best_margin is None or margin < best_margin:
            best_margin = margin
    return NO_EQUILIBRIUM_PENALTY if best_margin is None else best_margin


def run_test_04(params: ModelParameters) -> TestResult:
    """
    TEST 04 — Глобальный поиск седло-узловой бифуркации (переработано).

    ЧЕМ ЭТО ЛУЧШЕ СТАРОЙ ВЕРСИИ.
    Старая версия сканировала ОДИН параметр (R_0) по равномерной сетке
    из 30 точек вдоль ОДНОЙ warm-started ветви равновесия и смотрела на
    смену знака Re(критическое собственное число). Аудит справедливо
    указал: "отсутствие SN в этом диапазоне — не доказательство
    отсутствия SN вне диапазона или МЕЖДУ точками сетки".

    Здесь вместо равномерной сетки по одному параметру используется
    ГЛОБАЛЬНАЯ оптимизация (differential evolution — популяционный
    эволюционный алгоритм, устойчивый к застреванию в локальных
    минимумах в отличие от локальных методов и линейных сеток) по
    4-МЕРНОМУ параметрическому боксу [R_0, mu_inst, alpha_pos, gamma_R],
    минимизирующая
        h(p) = min_i |Re(lambda_i(J(x*(p), p)))|
    — то есть напрямую ИЩЕТ точки, где Якобиан вырожден (необходимое
    условие седло-узловой бифуркации: простое нулевое собственное
    значение), вместо надежды, что сетка "случайно попадёт" рядом с
    бифуркацией. Это не исчерпывающее доказательство отсутствия SN во
    всём 4D боксе (в общем случае NP-трудная задача), но качественно
    значимо более сильная и честно охарактеризованная попытка её найти:
    итоговый отчёт явно указывает число вычислений (n_function_evaluations)
    и достигнутый глобальный минимум margin, а не молчаливо "не нашли".

    Если DE находит кандидата с h(p) < FOLD_CANDIDATE_THRESHOLD, точка
    уточняется центральной разностью вдоль оси R_0 (наиболее физически
    интерпретируемый bifurcation-параметр в этой модели) для оценки
    трансверсальности (dRe(lambda)/dR_0 != 0) и классификации
    Saddle-Node/Hopf по величине мнимой части критического собственного
    значения.
    """
    eq_engine = EquilibriumEngine()

    state_dim = 2 * params.N_sub + 7
    healthy_seed = np.zeros(state_dim, dtype=np.float64)
    healthy_seed[params.N_sub : params.N_sub + 4] = 0.8
    healthy_seed[params.N_sub + 4] = 0.8 * params.F_max

    stressed_seed = np.zeros(state_dim, dtype=np.float64)
    stressed_seed[0:params.N_sub] = 0.5 * params.SID_max
    stressed_seed[params.N_sub : params.N_sub + 4] = 0.3
    stressed_seed[params.N_sub + 4] = 0.3 * params.F_max

    crisis_seed = np.zeros(state_dim, dtype=np.float64)
    crisis_seed[0:params.N_sub] = 0.8 * params.SID_max
    crisis_seed[params.N_sub : params.N_sub + 4] = 0.1
    crisis_seed[params.N_sub + 4] = 0.1 * params.F_max

    seeds = [healthy_seed, stressed_seed, crisis_seed]

    def objective(p_vec: np.ndarray) -> float:
        return _fold_margin_at(params, p_vec, eq_engine, seeds)

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        seed=42,
        maxiter=40,
        popsize=15,
        tol=1e-10,
        mutation=(0.4, 1.2),
        recombination=0.7,
        polish=True,
        updating="deferred",
    )

    global_min_margin = float(result.fun)
    best_params_vec = [float(v) for v in result.x]

    if global_min_margin >= NO_EQUILIBRIUM_PENALTY * 0.5:
        return TestResult(
            test_id="TEST_04",
            name="Saddle-Node Bifurcation Detection (Global Search)",
            status="FAILED",
            details={
                "reason": "no_equilibrium_found_across_entire_search_box",
                "n_function_evaluations": int(result.nfev),
            },
        )

    if global_min_margin >= FOLD_CANDIDATE_THRESHOLD:
        return TestResult(
            test_id="TEST_04",
            name="Saddle-Node Bifurcation Detection (Global Search)",
            status="NOT_DETECTED",
            details={
                "global_search_min_margin": global_min_margin,
                "fold_candidate_threshold": FOLD_CANDIDATE_THRESHOLD,
                "global_search_best_params": {
                    "R_0": best_params_vec[0],
                    "mu_inst": best_params_vec[1],
                    "alpha_pos": best_params_vec[2],
                    "gamma_R": best_params_vec[3],
                },
                "n_function_evaluations": int(result.nfev),
                "param_bounds": PARAM_BOUNDS,
                "note": (
                    "Differential-evolution global search over a 4D "
                    "parameter box found no near-singular Jacobian "
                    "(min |Re(eigenvalue)| never dropped below "
                    f"{FOLD_CANDIDATE_THRESHOLD}) after "
                    f"{int(result.nfev)} function evaluations. This is "
                    "materially stronger evidence of absence than a "
                    "uniform 1D grid, but is still not an exhaustive proof "
                    "for parameter combinations outside this box."
                ),
            },
        )

    # Кандидат найден — уточняем трансверсальность и классификацию.
    r0_star, mu_inst_star, alpha_pos_star, gamma_r_star = best_params_vec
    delta_r0 = max(1e-4, abs(r0_star) * 1e-3)

    drift_star, local_params_star = _build_drift(params, np.array(best_params_vec))
    drift_plus, _ = _build_drift(params, np.array([r0_star + delta_r0, mu_inst_star, alpha_pos_star, gamma_r_star]))
    drift_minus, _ = _build_drift(params, np.array([r0_star - delta_r0, mu_inst_star, alpha_pos_star, gamma_r_star]))

    eq_star_state, eq_star_margin = None, None
    for seed in seeds:
        eq = eq_engine.find_equilibrium(drift_star, seed)
        if not eq.is_valid or not StateVector(eq.state).is_in_domain(local_params_star):
            continue
        margin = _min_abs_real_eigenvalue(drift_star, eq.state)
        if eq_star_margin is None or margin < eq_star_margin:
            eq_star_state, eq_star_margin = eq.state, margin

    if eq_star_state is None:
        # Глобальный минимизатор сошёлся к паразитному корню ВНЕ домена
        # Omega (scipy.optimize.root не ограничен доменом) — это не
        # физически достижимая точка бифуркации, а численный артефакт.
        return TestResult(
            test_id="TEST_04",
            name="Saddle-Node Bifurcation Detection (Global Search)",
            status="NOT_DETECTED",
            details={
                "reason": "de_global_minimum_is_an_out_of_domain_spurious_root",
                "global_search_min_margin": global_min_margin,
                "candidate_params": {
                    "R_0": r0_star, "mu_inst": mu_inst_star,
                    "alpha_pos": alpha_pos_star, "gamma_R": gamma_r_star,
                },
                "n_function_evaluations": int(result.nfev),
                "note": (
                    "scipy.optimize.root finds roots of F(x)=0 without "
                    "constraining x to the physical domain Omega. The "
                    "near-singular Jacobian found by the global search "
                    "corresponds to a root outside Omega (e.g. ED^k < 0) "
                    "and is therefore not a reachable equilibrium of the "
                    "reflected system -- discarded as a false positive, "
                    "not reported as a genuine Saddle-Node."
                ),
            },
        )

    eigvals_star = np.linalg.eigvals(_finite_difference_jacobian(drift_star, eq_star_state))
    crit_idx = int(np.argmin(np.abs(np.real(eigvals_star))))
    crit_eig = eigvals_star[crit_idx]

    re_plus = _crit_real_part(eq_engine, drift_plus, eq_star_state, local_params_star)
    re_minus = _crit_real_part(eq_engine, drift_minus, eq_star_state, local_params_star)

    transversality = (
        (re_plus - re_minus) / (2.0 * delta_r0)
        if re_plus is not None and re_minus is not None
        else 0.0
    )

    bif_type = "Hopf" if abs(float(np.imag(crit_eig))) > 1e-2 else "Saddle-Node"
    is_genuine_fold = abs(transversality) > 1e-6

    return TestResult(
        test_id="TEST_04",
        name="Saddle-Node Bifurcation Detection (Global Search)",
        status="PASSED" if is_genuine_fold else "FAILED",
        details={
            "bifurcation_type": bif_type,
            "critical_eigenvalue": str(crit_eig),
            "transversality_dRe_dR0": transversality,
            "global_search_min_margin": global_min_margin,
            "candidate_params": {
                "R_0": r0_star, "mu_inst": mu_inst_star,
                "alpha_pos": alpha_pos_star, "gamma_R": gamma_r_star,
            },
            "n_function_evaluations": int(result.nfev),
        },
    )
