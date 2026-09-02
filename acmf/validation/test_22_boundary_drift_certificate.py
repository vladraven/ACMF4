import numpy as np
from config.schema import (
    ModelParametersSchema,
    SystemDimensionsConfig,
    DecayAndRatesConfig,
    MetabolismConfig,
    EpistemicAndReformConfig,
    ContagionAndHawkesConfig,
)
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.model.forcing import ForcingState
from acmf.model.dynamics import compute_full_drift_vector
from acmf.validation.result import TestResult

# Число независимых случайных испытаний (параметры + состояние + forcing)
# на КАЖДУЮ границу КАЖДОЙ переменной. Итоговое число вычислений полного
# вектора дрейфа — N_TRIALS_PER_BOUNDARY * (число проверяемых границ).
N_TRIALS_PER_BOUNDARY = 400

# Допуск на погрешность плавающей точки при сравнении дрейфа с нулём.
TOL = 1e-9


def _sample_perturbed_params(rng: np.random.Generator) -> ModelParameters:
    """
    Строит случайный, но ВСЕГДА валидный (относительно pydantic-схемы)
    набор параметров модели, варьируя базовые значения в кратности
    [0.4x .. 2.5x] (а знако-определённые/пороговые параметры — в своих
    естественных диапазонах). Это даёт покрытие операционного конверта
    параметров (operational envelope), а не одну фиксированную точку.
    """
    def u(lo: float, hi: float) -> float:
        return float(rng.uniform(lo, hi))

    schema = ModelParametersSchema(
        dimensions=SystemDimensionsConfig(
            F_max=u(0.4, 2.5), SID_buf=u(0.4, 2.5), SID_max=u(0.4, 2.5),
            kappa_s=u(1.0, 20.0),
        ),
        dynamics=DecayAndRatesConfig(
            alpha_pos=u(0.0, 2.5), beta_neg=u(0.0, 2.5), gamma_inst=u(0.0, 2.5),
            mu_inst=u(0.0, 2.5), alpha_F=u(0.0, 2.5), beta_F=u(0.0, 2.5),
            alpha_Ch=u(0.0, 2.5), beta_Ch=u(0.0, 2.5), mu_Ch=u(0.0, 2.5),
            alpha_Prod=u(0.0, 2.5), beta_Prod=u(0.0, 2.5), alpha_M=u(0.0, 2.5),
            mu_M=u(0.0, 2.5), gamma_scar=u(0.0, 2.5), mu_scar=u(0.0, 2.5),
            Threshold_scar=u(-1.0, 1.0), gamma_R=u(0.0, 1.0),
            theta_A=u(0.0, 2.5), theta_P=u(0.0, 2.5), theta_I=u(0.0, 2.5),
        ),
        metabolism=MetabolismConfig(
            w=[[u(0.0, 1.0) for _ in range(3)] for _ in range(3)],
            p=[[u(0.0, 1.0) for _ in range(2)] for _ in range(3)],
            eta=u(0.0, 2.0),
            Capacity=[u(0.1, 2.0) for _ in range(3)],
            rho=[u(0.0, 1.0) for _ in range(3)],
            sigma_0=[u(0.0, 1.0) for _ in range(3)],
        ),
        epistemic=EpistemicAndReformConfig(
            # alpha_mask документально трактуется как доля в [0,1]
            # (компонента VisibilityGap = alpha_mask*(1-V)*(1-G) должна
            # оставаться в [0,1]); схема этого явно не валидирует, но
            # операционный конверт параметров ограничен этим диапазоном.
            alpha_mask=[u(0.0, 1.0) for _ in range(3)],
            lambda_burst=u(0.0, 2.5), alpha_burst=u(0.1, 10.0),
            ED_crit=u(0.0, 2.0), ED_scale=u(0.1, 2.0), ED_impact=u(0.0, 2.0),
            RefThresh=u(-1.0, 1.0), lambda_ref_0=u(0.0, 2.5),
            omega_fatigue=u(0.0, 1.0), tau_ref=u(0.0, 2.5),
            Delta_t=u(0.0, 1.0), Delta_ref=u(0.0, 1.0),
        ),
        contagion=ContagionAndHawkesConfig(
            kappa_spill=u(0.0, 1.0), SID_contagion=u(-1.0, 1.0), beta_H=u(0.5, 3.0),
            Gamma=[[0.05, 0.02, 0.02], [0.02, 0.05, 0.02], [0.02, 0.02, 0.05]],
            mu_rec=u(0.0, 2.5), omega_V=u(0.0, 1.0), omega_SID=u(0.0, 1.0),
        ),
    )
    return ModelParameters.from_schema(schema)


def _random_forcing(rng: np.random.Generator) -> ForcingState:
    return ForcingState(
        A=float(rng.uniform(0.0, 3.0)),
        R_0=float(rng.uniform(0.0, 3.0)),
        V=float(rng.uniform(0.0, 1.0)),
        G=float(rng.uniform(0.0, 1.0)),
        dA_dt=float(rng.uniform(-2.0, 2.0)),
    )


def _random_interior_state(rng: np.random.Generator, params: ModelParameters) -> np.ndarray:
    dim = 2 * params.N_sub + 7
    x = np.zeros(dim, dtype=np.float64)
    x[0:params.N_sub] = rng.uniform(-params.SID_buf, params.SID_max, params.N_sub)
    x[params.N_sub : params.N_sub + 4] = rng.uniform(0.0, 1.0, 4)
    x[params.N_sub + 4] = rng.uniform(0.0, params.F_max)
    x[params.N_sub + 5] = rng.uniform(0.0, 1.0)
    x[params.N_sub + 6 : 2 * params.N_sub + 6] = rng.uniform(0.0, 5.0, params.N_sub)
    x[2 * params.N_sub + 6] = rng.uniform(0.0, 5.0)
    return x


def _drift_at(params: ModelParameters, x: np.ndarray, forcing: ForcingState, rng: np.random.Generator) -> np.ndarray:
    st = StateVector(x)
    return compute_full_drift_vector(
        st,
        forcing,
        delayed_d_a_dt=float(rng.uniform(-2.0, 2.0)),
        delayed_d_prod_dt=float(rng.uniform(-2.0, 2.0)),
        delayed_d_inst_dt=float(rng.uniform(-2.0, 2.0)),
        delayed_d_agg_sid_obs_dt=float(rng.uniform(-2.0, 2.0)),
        spillover=np.zeros(params.N_sub, dtype=np.float64),
        params=params,
    )


def run_test_22(base_params: ModelParameters) -> TestResult:
    """
    TEST 22 — Сертификат направленного внутрь дрейфа (Inward-Pointing
    Drift Certificate) на границах домена Omega.

    ЧТО ДОКАЗЫВАЕТСЯ И ПОЧЕМУ.
    Документ ACMF (§17) требует явного доказательства, что дрейф на
    границах ограниченных переменных всегда направлен внутрь домена —
    то есть что оператор отражения Скорохода (TEST_01) НЕ ЯВЛЯЕТСЯ
    единственной причиной, по которой траектория не покидает Omega, а
    является лишь защитным механизмом от дискретизационной ошибки
    Эйлера. До этого теста утверждение "inward-pointing drift
    предполагается" было АКСИОМОЙ без проверки.

    Диффузия (compute_diffusion_sigma) действует ТОЛЬКО на SID[0:3], а
    скачки (jump_generator) — тоже только на SID[0:3] (см.
    euler_maruyama.py: diffusion_term/jump_term ненулевые только в
    диапазоне [0:3]). Поэтому для восьми переменных: Inst, Ch, Prod, M,
    F, Scar, ED^k (k=0..N_sub-1), RecDebt — единственный источник
    движения это детерминированный дрейф F(X). Если для каждой из этих
    переменных дрейф на нижней границе >= 0, а на верхней границе <= 0
    (при ЛЮБЫХ допустимых параметрах и ЛЮБОМ допустимом состоянии
    остальных координат), то ТРАЕКТОРИЯ МАТЕМАТИЧЕСКИ НЕ МОЖЕТ покинуть
    домен по этим осям — и отражение Скорохода в reflection.py для них
    является чистым запасом на ошибку дискретизации, а не
    компенсацией реального выхода из Omega.

    Аналитическая лемма (проверяется здесь численно на плотной
    случайной выборке операционного конверта параметров/состояний, а
    не в одной точке):
      - s_plus(x, kappa) = log(1+exp(kappa*x))/kappa > 0 для kappa>0
        при ЛЮБОМ x (numpy.logaddexp(0, kappa*x) = log(1+exp(kappa*x))
        строго больше log(1) = 0). Отсюда каждое слагаемое вида
        beta*s_plus(...) с beta>=0 (все "beta/mu/alpha/gamma"-параметры
        схема валидирует как >=0) неотрицательно.
      - Inst: d/dt|_{Inst=0} = inst_growth >= 0 (сумма неотрицательных
        множителей: R_eff=R_0*(1-gamma_R*Scar)>=0 при R_0>=0,
        gamma_R,Scar in [0,1]; Ch,M,G in [0,1]; reform_impulse>=0 как
        произведение неотрицательных awareness/fatigue/G/delay_supp).
        d/dt|_{Inst=1} = -inst_decay <= 0 (mu_inst>=0, beta_neg>=0,
        s_plus>=0).
      - Ch, Prod, M, F, Scar — симметричные по структуре леммы
        (см. compute_bounded_ode_drifts): на нижней границе остаётся
        только неотрицательное "growth"-слагаемое, на верхней —
        только неположительное "decay"-слагаемое.
      - ED^k при ED^k=0: d_ED^k = W_true^k - W_obs^k =
        W_true^k * VisibilityGap^k >= 0, т.к. W_true>=0 (клэмп
        np.maximum(0, ...) в compute_true_wear) и VisibilityGap =
        alpha_mask*(1-V)*(1-G) >= 0 при alpha_mask, V, G в [0,1].
        Верхней границы у ED нет (домен [0, +inf)).
      - RecDebt при RecDebt=0: d_RecDebt = -mu_rec * 0 = 0 ТОЧНО
        (алгебраическое тождество, mu_rec >= 0) — граница строго
        неотразима по построению, без всякой аппроксимации.

    Тест ПАДАЕТ, если для любой из N_TRIALS_PER_BOUNDARY случайных
    комбинаций (параметры + состояние + forcing + запаздывающие
    производные) знак дрейфа на границе нарушает лемму — это означало
    бы либо ошибку в реализации compute_full_drift_vector, либо
    нарушение самой леммы за пределами предполагаемого операционного
    конверта параметров.
    """
    rng = np.random.default_rng(20260902)
    N_sub = base_params.N_sub

    violations: list[dict] = []
    worst_margin = float("inf")
    n_checks = 0

    # Индексы и метаданные границ: (label, index, bound_value, ожидаемый_знак)
    # знак "+1" значит "дрейф должен быть >= 0" (нижняя граница, вход внутрь),
    # знак "-1" значит "дрейф должен быть <= 0" (верхняя граница, вход внутрь).
    scalar_boundaries = [
        ("Inst_lower", N_sub, 0.0, +1),
        ("Inst_upper", N_sub, 1.0, -1),
        ("Ch_lower", N_sub + 1, 0.0, +1),
        ("Ch_upper", N_sub + 1, 1.0, -1),
        ("Prod_lower", N_sub + 2, 0.0, +1),
        ("Prod_upper", N_sub + 2, 1.0, -1),
        ("M_lower", N_sub + 3, 0.0, +1),
        ("M_upper", N_sub + 3, 1.0, -1),
        ("Scar_lower", N_sub + 5, 0.0, +1),
        ("Scar_upper", N_sub + 5, 1.0, -1),
        ("RecDebt_lower", 2 * N_sub + 6, 0.0, +1),
    ]

    for label, idx, bound_val, sign in scalar_boundaries:
        for _ in range(N_TRIALS_PER_BOUNDARY):
            params = _sample_perturbed_params(rng)
            state_dim = 2 * params.N_sub + 7
            if idx >= state_dim:
                continue
            x = _random_interior_state(rng, params)
            x[idx] = bound_val
            forcing = _random_forcing(rng)
            drift = _drift_at(params, x, forcing, rng)
            value = float(drift[idx])
            margin = sign * value  # должно быть >= -TOL
            n_checks += 1
            worst_margin = min(worst_margin, margin)
            if margin < -TOL:
                violations.append({"boundary": label, "drift_value": value, "margin": margin})

    # F имеет отдельную границу [0, F_max] с переменным F_max
    for label, bound_frac, sign in [("F_lower", 0.0, +1), ("F_upper", 1.0, -1)]:
        for _ in range(N_TRIALS_PER_BOUNDARY):
            params = _sample_perturbed_params(rng)
            idx = params.N_sub + 4
            x = _random_interior_state(rng, params)
            x[idx] = bound_frac * params.F_max
            forcing = _random_forcing(rng)
            drift = _drift_at(params, x, forcing, rng)
            value = float(drift[idx])
            margin = sign * value
            n_checks += 1
            worst_margin = min(worst_margin, margin)
            if margin < -TOL:
                violations.append({"boundary": label, "drift_value": value, "margin": margin})

    # ED^k, k=0..N_sub-1, только нижняя граница (домен не ограничен сверху)
    for _ in range(N_TRIALS_PER_BOUNDARY):
        params = _sample_perturbed_params(rng)
        for k in range(params.N_sub):
            idx = params.N_sub + 6 + k
            x = _random_interior_state(rng, params)
            x[idx] = 0.0
            forcing = _random_forcing(rng)
            drift = _drift_at(params, x, forcing, rng)
            value = float(drift[idx])
            margin = value  # должно быть >= -TOL (нижняя граница)
            n_checks += 1
            worst_margin = min(worst_margin, margin)
            if margin < -TOL:
                violations.append({"boundary": f"ED_{k}_lower", "drift_value": value, "margin": margin})

    status = "PASSED" if len(violations) == 0 else "FAILED"

    return TestResult(
        test_id="TEST_22",
        name="Boundary Inward-Pointing Drift Certificate",
        status=status,
        details={
            "n_boundary_checks": n_checks,
            "n_violations": len(violations),
            "worst_case_margin": worst_margin,
            "first_violations": violations[:5],
            "scope": (
                "Certifies Inst, Ch, Prod, M, F, Scar, ED^k, RecDebt — the "
                "8 variable families with ZERO diffusion/jump term (only "
                "SID[0:3] receives noise and jumps per euler_maruyama.py). "
                "SID boundary behaviour is deliberately NOT claimed to be "
                "drift-inward (spillover/jumps can push it outward by "
                "design) and remains the responsibility of the Skorokhod "
                "reflector validated in TEST_01."
            ),
        },
    )
