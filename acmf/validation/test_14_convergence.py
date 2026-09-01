import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingProfile
from acmf.model.dynamics import compute_full_drift_vector
from acmf.stochastic.diffusion import compute_diffusion_sigma
from acmf.solver.base import SolverDomain
from acmf.solver.engine import ACMFEngine
from acmf.validation.result import TestResult

# Относительный, а не абсолютный порог для стандартной ошибки среднего.
# Абсолютное значение 0.1 не имело обоснования и не масштабировалось
# с характерным диапазоном самой переменной Inst — при другом наборе
# параметров (другой mean_inst) тот же порог мог быть как избыточно
# строгим, так и пропускать реально нестабильную MC-оценку.
RELATIVE_SE_TOLERANCE = 0.05

# Нижний предел масштаба в знаменателе — защита от деления на
# величину, близкую к нулю, если mean_inst окажется около 0.
MIN_SCALE_FLOOR = 0.05


def run_test_14(params: ModelParameters) -> TestResult:
    """TEST 14 — Монте-Карло сходимость статистических моментов траектории."""
    domain = SolverDomain(sid_buf=params.SID_buf, sid_max=params.SID_max, f_max=params.F_max)
    engine = ACMFEngine(domain=domain, scheme="euler_maruyama")
    forcing = ForcingProfile().evaluate(0.0)

    # Размерность состояния выводится из контракта модели (9 базовых
    # компонент + N_sub подсистем SID + 1), а не захардкожена как 13 —
    # ранее это расходилось со стилем test_05/test_21 и было хрупко
    # при изменении N_sub.
    state_dim = 9 + params.N_sub + 1
    init_state = np.zeros(state_dim, dtype=np.float64)
    init_state[3:7] = 0.8
    init_state[7] = 0.8 * params.F_max

    def drift_fn(x, d_a, d_p, d_i, d_agg):
        st = StateVector(x)
        return compute_full_drift_vector(
            st, forcing, d_a, d_p, d_i, d_agg, np.zeros(params.N_sub), params
        )

    def diff_fn(x):
        st = StateVector(x)
        return compute_diffusion_sigma(st, forcing, params)

    n_runs = 20
    final_inst_values = []

    for seed in range(n_runs):
        traj = engine.simulate(
            initial_state=init_state,
            t_span=(0.0, 10.0),
            dt=0.05,
            drift_fn=drift_fn,
            diffusion_fn=diff_fn,
            random_seed=seed,
        )
        final_inst_values.append(traj.states[-1, 3])

    mean_inst = float(np.mean(final_inst_values))
    std_err = float(np.std(final_inst_values) / np.sqrt(n_runs))

    scale = max(abs(mean_inst), MIN_SCALE_FLOOR)
    relative_error = std_err / scale
    is_converged = relative_error < RELATIVE_SE_TOLERANCE

    status = "PASSED" if is_converged else "FAILED"

    return TestResult(
        test_id="TEST_14",
        name="Monte Carlo Moment Convergence",
        status=status,
        details={
            "n_runs": n_runs,
            "mean_inst": mean_inst,
            "std_error": std_err,
            "relative_error": relative_error,
            "relative_se_tolerance": RELATIVE_SE_TOLERANCE,
        },
    )
