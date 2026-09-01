import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult

# Минимальный допустимый порядок эмпирической сходимости (log-log
# наклон между дискретизациями). EM имеет теоретический strong order
# 0.5, Milstein — 1.0; сравниваем траектории двух схем друг с другом
# (а не с общим точным решением), поэтому наблюдаемый порядок этой
# разности — величина, ограниченная снизу более слабой схемой (EM),
# но с меньшим запасом из-за шумовой природы SDE. Порог занижен
# намеренно, чтобы не давать ложных FAIL из-за стохастического шума.
MIN_ACCEPTABLE_CONVERGENCE_ORDER = 0.15

# Минимальный множитель падения ошибки между самым грубым и самым
# мелким шагом — защита от случая, когда log-log наклон формально
# положителен, но эффект пренебрежимо мал (шум/округление).
MIN_ERROR_REDUCTION_FACTOR = 1.5


def run_test_21(params: ModelParameters) -> TestResult:
    """
    TEST 21 — Сходимость и независимость Euler-Maruyama/Milstein.

    ВАЖНОЕ ОГРАНИЧЕНИЕ ДОСТОВЕРНОСТИ (документируется явно, а не
    скрывается): ACMFEngine.simulate принимает только random_seed, а
    не общий поток приращений Винеровского процесса (dW). Поэтому
    траектории при разных dt для одного и того же random_seed не
    гарантированно используют согласованные (matched) реализации
    броуновского пути — генератор с тем же seed при разном числе
    шагов расходует случайные числа по-другому. Из-за этого нельзя
    утверждать строгий Richardson-style порядок сходимости в
    классическом смысле. Тест ниже — эмпирическая проверка на монотонный
    и количественно значимый тренд убывания расхождения EM/Milstein
    при измельчении шага, а не доказательство теоретического порядка.
    Для полной строгости здесь нужен API солвера с инжектируемым
    общим dW-потоком (см. ACMF4 audit item про engine.py).
    """
    domain = SolverDomain(
        sid_buf=params.SID_buf,
        sid_max=params.SID_max,
        f_max=params.F_max,
    )
    forcing = ForcingProfile().evaluate(0.0)

    init_state = np.zeros(
        9 + params.N_sub + 1,
        dtype=np.float64,
    )
    init_state[3:7] = 0.5

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st,
            forcing,
            d_a,
            d_p,
            d_i,
            d_agg,
            np.zeros(params.N_sub),
            params,
        )

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(
            st,
            forcing,
            params,
        )

    dt_values = (0.04, 0.02, 0.01)
    discrepancies = []

    for dt in dt_values:
        engine_em = ACMFEngine(
            domain=domain,
            scheme="euler_maruyama",
        )
        engine_mil = ACMFEngine(
            domain=domain,
            scheme="milstein",
        )

        traj_em = engine_em.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=42,
        )
        traj_mil = engine_mil.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=dt,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=42,
        )

        if traj_em.states.shape != traj_mil.states.shape:
            return TestResult(
                test_id="TEST_21",
                name="Solver Independence Convergence",
                status="FAILED",
                details={"reason": "trajectory_shape_mismatch"},
            )

        discrepancies.append(
            float(
                np.max(
                    np.abs(
                        traj_em.states
                        - traj_mil.states
                    )
                )
            )
        )

    # 1. Строгая монотонность по ВСЕМ последовательным парам, а не
    # только по конечным точкам (было: discrepancies[-1] <= discrepancies[0]).
    is_monotonic = all(
        discrepancies[i + 1] <= discrepancies[i]
        for i in range(len(discrepancies) - 1)
    )

    # 2. Эмпирический порядок сходимости через log-log наклон
    # (линейная регрессия по всем трём точкам, не по двум крайним).
    log_dt = np.log(np.array(dt_values, dtype=np.float64))
    log_err = np.log(
        np.maximum(np.array(discrepancies, dtype=np.float64), 1e-300)
    )
    slope, _intercept = np.polyfit(log_dt, log_err, 1)
    observed_order = float(slope)

    # 3. Абсолютное падение ошибки от самого грубого к самому мелкому
    # шагу должно быть содержательным, а не в пределах шума.
    error_reduction_factor = (
        discrepancies[0] / discrepancies[-1]
        if discrepancies[-1] > 0.0
        else float("inf")
    )

    order_ok = observed_order >= MIN_ACCEPTABLE_CONVERGENCE_ORDER
    reduction_ok = error_reduction_factor >= MIN_ERROR_REDUCTION_FACTOR

    converges = is_monotonic and order_ok and reduction_ok

    return TestResult(
        test_id="TEST_21",
        name="Solver Independence Convergence",
        status="PASSED" if converges else "FAILED",
        details={
            "dt_values": dt_values,
            "max_discrepancies": discrepancies,
            "is_monotonic_all_pairs": is_monotonic,
            "observed_convergence_order": observed_order,
            "min_acceptable_order": MIN_ACCEPTABLE_CONVERGENCE_ORDER,
            "error_reduction_factor": error_reduction_factor,
            "min_required_reduction_factor": MIN_ERROR_REDUCTION_FACTOR,
            "caveat": (
                "EM/Milstein trajectories at different dt are not "
                "guaranteed to share matched Brownian increments for the "
                "same random_seed; this is an empirical trend check, not "
                "a strict Richardson convergence-order proof."
            ),
        },
    )
