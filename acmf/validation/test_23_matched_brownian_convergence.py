import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.reflection import SkorokhodReflector
from acmf.solver.euler_maruyama import EulerMaruyamaStep
from acmf.validation.result import TestResult

# Число независимых траекторий Монте-Карло на каждый шаг dt (усреднение
# сильной ошибки, чтобы устранить влияние конкретной реализации пути).
N_PATHS = 12

# Самый мелкий шаг, задающий "эталонный" путь и являющийся ИСТОЧНИКОМ
# приращений броуновского пути для всех остальных dt (см. docstring).
DT_FINE = 0.00125

DT_VALUES = (0.02, 0.01, 0.005)

T_END = 4.0

MIN_ACCEPTABLE_CONVERGENCE_ORDER = 0.3
MIN_ERROR_REDUCTION_FACTOR = 1.5


def run_test_23(params: ModelParameters) -> TestResult:
    """
    TEST 23 — Строгая (matched-path) сходимость Эйлера-Маруямы.

    ПОЧЕМУ ЭТОТ ТЕСТ НУЖЕН (закрывает явный caveat TEST_21).
    TEST_21 честно документирует ограничение: ACMFEngine.simulate
    принимает только random_seed, а не общий поток приращений
    Винеровского процесса (dW), поэтому траектории EM/Milstein при
    разных dt для одного seed НЕ используют согласованные реализации
    броуновского пути — это делает наблюдаемый в TEST_21 "порядок
    сходимости" эмпирическим трендом, а не строгим доказательством.

    ЗДЕСЬ мы строим ОДИН фиксированный путь броуновского движения на
    самой мелкой сетке (dt_fine) генератором dW_fine ~ N(0, dt_fine),
    а затем получаем СОГЛАСОВАННЫЕ приращения для каждого более
    грубого шага dt = k*dt_fine суммированием k последовательных
    мелких приращений:
        dW_coarse[i] = sum_{j=i*k}^{(i+1)*k-1} dW_fine[j]
    Это математически ТОЧНО (не приближённо) соответствует одному и
    тому же непрерывному пути W(t), просто наблюдаемому на разных
    сетках — фундаментальное свойство независимых приращений
    броуновского движения. Это устраняет ровно тот дефект, который
    делает TEST_21 нестрогим.

    Для каждого пути и каждого dt интегрируем схему Эйлера-Маруямы
    (с тем же оператором отражения Скорохода, что и в проде) и
    сравниваем полученную траекторию с "эталонной" траекторией на
    dt_fine В ОБЩИХ узлах сетки. Порядок сходимости оценивается по
    log-log наклону среднего по N_PATHS путям максимального отклонения
    от эталона. Теоретический сильный порядок EM для гладкой SDE без
    отражения — 0.5; нелинейность отражения на границе и негладкость
    s_plus/s_minus дают законное основание ожидать несколько меньший
    эмпирический порядок, поэтому порог занижен но обоснован (0.3),
    как и в TEST_21, — однако, в отличие от TEST_21, здесь это
    ДЕЙСТВИТЕЛЬНО Richardson-style сравнение с общим путём, а не
    сравнение двух независимых реализаций шума.
    """
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    forcing = ForcingProfile().evaluate(0.0)
    reflector = SkorokhodReflector(domain)
    stepper = EulerMaruyamaStep()

    state_dim = 2 * params.N_sub + 7
    init_state = np.zeros(state_dim, dtype=np.float64)
    init_state[params.N_sub : params.N_sub + 4] = 0.5

    def drift_fn(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, 0.0, 0.0, 0.0, 0.0, np.zeros(params.N_sub), params
        )

    def diff_fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    n_fine_steps = int(round(T_END / DT_FINE))

    def integrate(dt: float, dw_steps: np.ndarray) -> np.ndarray:
        """Интегрирует один путь с заданными (уже согласованными) приращениями dW на шаге dt."""
        state = init_state.copy()
        n_steps = dw_steps.shape[0]
        traj = np.zeros((n_steps + 1, state_dim), dtype=np.float64)
        traj[0] = state
        for i in range(n_steps):
            drift = drift_fn(state)
            sigma, d_sigma = diff_fn(state)
            random_normal = dw_steps[i] / np.sqrt(dt)
            raw_next = stepper.step(
                current_state=state,
                drift=drift,
                diffusion_sigma=sigma,
                diffusion_derivative=d_sigma,
                random_normal=random_normal,
                jump_vector=np.zeros(params.N_sub, dtype=np.float64),
                dt=dt,
            )
            state, _diag = reflector.reflect_state(raw_next)
            traj[i + 1] = state
        return traj

    per_dt_errors: dict[float, list[float]] = {dt: [] for dt in DT_VALUES}

    for path_idx in range(N_PATHS):
        rng = np.random.default_rng(9000 + path_idx)
        dw_fine = rng.standard_normal((n_fine_steps, params.N_sub)) * np.sqrt(DT_FINE)

        ref_traj = integrate(DT_FINE, dw_fine)

        for dt in DT_VALUES:
            ratio = int(round(dt / DT_FINE))
            assert abs(ratio * DT_FINE - dt) < 1e-9, "dt must be an exact multiple of DT_FINE"
            n_coarse_steps = n_fine_steps // ratio
            dw_coarse = dw_fine[: n_coarse_steps * ratio].reshape(n_coarse_steps, ratio, params.N_sub).sum(axis=1)

            coarse_traj = integrate(dt, dw_coarse)

            # Сравнение в ОБЩИХ узлах времени: coarse-шаг i соответствует
            # fine-шагу i*ratio.
            ref_at_common = ref_traj[0 : n_coarse_steps * ratio + 1 : ratio]
            max_dev = float(np.max(np.abs(coarse_traj - ref_at_common)))
            per_dt_errors[dt].append(max_dev)

    mean_errors = [float(np.mean(per_dt_errors[dt])) for dt in DT_VALUES]

    is_monotonic = all(
        mean_errors[i + 1] <= mean_errors[i] for i in range(len(mean_errors) - 1)
    )

    log_dt = np.log(np.array(DT_VALUES, dtype=np.float64))
    log_err = np.log(np.maximum(np.array(mean_errors, dtype=np.float64), 1e-300))
    slope, _intercept = np.polyfit(log_dt, log_err, 1)
    observed_order = float(slope)

    error_reduction_factor = (
        mean_errors[0] / mean_errors[-1] if mean_errors[-1] > 0.0 else float("inf")
    )

    order_ok = observed_order >= MIN_ACCEPTABLE_CONVERGENCE_ORDER
    reduction_ok = error_reduction_factor >= MIN_ERROR_REDUCTION_FACTOR
    converges = is_monotonic and order_ok and reduction_ok

    return TestResult(
        test_id="TEST_23",
        name="Matched-Brownian-Path Strong Convergence",
        status="PASSED" if converges else "FAILED",
        details={
            "dt_fine": DT_FINE,
            "dt_values": DT_VALUES,
            "n_paths": N_PATHS,
            "mean_max_deviation": mean_errors,
            "is_monotonic": is_monotonic,
            "observed_convergence_order": observed_order,
            "min_acceptable_order": MIN_ACCEPTABLE_CONVERGENCE_ORDER,
            "error_reduction_factor": error_reduction_factor,
            "min_required_reduction_factor": MIN_ERROR_REDUCTION_FACTOR,
            "note": (
                "Uses a single shared fine-grid Brownian path per Monte "
                "Carlo trial, with coarse dW obtained by exact summation "
                "of matching fine increments -- a true Richardson-style "
                "strong-convergence test against a common noise "
                "realisation, unlike TEST_21's independent-seed comparison."
            ),
        },
    )
