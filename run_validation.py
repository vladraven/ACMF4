import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from acmf.validation.framework import ValidationFramework
from acmf.validation.test_00_baseline import run_test_00
from acmf.validation.test_01_domain import run_test_01
from acmf.validation.test_02_equilibrium import run_test_02
from acmf.validation.test_03_stability import run_test_03
from acmf.validation.test_04_saddle_node import run_test_04
from acmf.validation.test_05_hopf import run_test_05
from acmf.validation.test_06_recovery import run_test_06
from acmf.validation.test_07_shock_threshold import run_test_07
from acmf.validation.test_08_hawkes import run_test_08
from acmf.validation.test_09_spatial import run_test_09
from acmf.validation.test_10_hybrid import run_test_10
from acmf.validation.test_11_ews_lead import run_test_11
from acmf.validation.test_12_fpr import run_test_12
from acmf.validation.test_13_fnr import run_test_13
from acmf.validation.test_14_convergence import run_test_14
from acmf.validation.test_15_sobol import run_test_15
from acmf.validation.test_16_feasibility import run_test_16
from acmf.validation.test_17_timescale import run_test_17
from acmf.validation.test_18_hysteresis import run_test_18
from acmf.validation.test_19_recovery_distribution import run_test_19
from acmf.validation.test_20_counterfactual import run_test_20
from acmf.validation.test_21_solver_independence import run_test_21


def build_default_parameters() -> ModelParameters:
    schema = ModelParametersSchema(
        dimensions=SystemDimensionsConfig(
            F_max=1.0, SID_buf=1.0, SID_max=3.0, kappa_s=10.0
        ),
        dynamics=DecayAndRatesConfig(
            alpha_pos=1.0, beta_neg=1.0, gamma_inst=0.2, mu_inst=0.1,
            alpha_F=0.5, beta_F=0.5, alpha_Ch=0.8, beta_Ch=0.4, mu_Ch=0.05,
            alpha_Prod=1.0, beta_Prod=0.5, alpha_M=0.5, mu_M=0.1,
            gamma_scar=0.3, mu_scar=0.02, Threshold_scar=0.5, gamma_R=0.5,
            theta_A=0.1, theta_P=0.1, theta_I=0.1,
        ),
        metabolism=MetabolismConfig(
            w=[[0.4, 0.4, 0.2], [0.5, 0.5, 0.0], [0.6, 0.4, 0.0]],
            p=[[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
            eta=0.1,
            Capacity=[1.0, 1.0, 1.0],
            rho=[0.2, 0.2, 0.2],
            sigma_0=[0.1, 0.1, 0.1],
        ),
        epistemic=EpistemicAndReformConfig(
            alpha_mask=[0.3, 0.3, 0.3], lambda_burst=1.0, alpha_burst=5.0,
            ED_crit=0.5, ED_scale=1.0, ED_impact=0.5, RefThresh=0.2,
            lambda_ref_0=0.5, omega_fatigue=0.5, tau_ref=1.0,
            Delta_t=0.5, Delta_ref=0.5,
        ),
        contagion=ContagionAndHawkesConfig(
            kappa_spill=0.1, SID_contagion=0.5, beta_H=2.0,
            Gamma=[[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]],
            mu_rec=0.1, omega_V=0.2, omega_SID=0.2,
        ),
    )
    return ModelParameters.from_schema(schema)


def main() -> None:
    print("=== ACMF 4.9.3.1 Full Comprehensive Validation Suite (TEST 00 - 21) ===")
    params = build_default_parameters()
    framework = ValidationFramework()

    # Уровень 1: Математическая корректность
    framework.register_test("TEST_00", lambda: run_test_00(params))
    framework.register_test("TEST_01", lambda: run_test_01(params))
    framework.register_test("TEST_16", lambda: run_test_16(params))
    framework.register_test("TEST_21", lambda: run_test_21(params))

    # Уровень 2: Динамическая устойчивость и бифуркации
    framework.register_test("TEST_02", lambda: run_test_02(params))
    framework.register_test("TEST_03", lambda: run_test_03(params))
    framework.register_test("TEST_04", lambda: run_test_04(params))
    framework.register_test("TEST_05", lambda: run_test_05(params))
    framework.register_test("TEST_06", lambda: run_test_06(params))
    framework.register_test("TEST_07", lambda: run_test_07(params))
    framework.register_test("TEST_09", lambda: run_test_09(params))
    framework.register_test("TEST_10", lambda: run_test_10(params))
    framework.register_test("TEST_17", lambda: run_test_17(params))
    framework.register_test("TEST_18", lambda: run_test_18(params))
    framework.register_test("TEST_20", lambda: run_test_20(params))

    # Уровень 3: Стохастика, EWS и глобальная чувствительность
    framework.register_test("TEST_08", lambda: run_test_08(params))
    framework.register_test("TEST_11", lambda: run_test_11(params))
    framework.register_test("TEST_12", lambda: run_test_12(params))
    framework.register_test("TEST_13", lambda: run_test_13(params))
    framework.register_test("TEST_14", lambda: run_test_14(params))
    framework.register_test("TEST_15", lambda: run_test_15(params))
    framework.register_test("TEST_19", lambda: run_test_19(params))

    results = framework.run_all()
    for res in results:
        # ИСПРАВЛЕНО: NOT_DETECTED — легитимный, отдельный от FAILED статус
        # (см. §TEST04 документа: "отсутствие Saddle-Node в выбранном
        # диапазоне не является ошибкой модели"). Раньше он визуально
        # схлопывался в "[✗] FAIL" наравне с настоящими провалами.
        if res.status == "PASSED":
            status_flag = "[✓] PASS"
        elif res.status == "NOT_DETECTED":
            status_flag = "[•] N/D "
        else:
            status_flag = "[✗] FAIL"
        print(f"{status_flag} | {res.test_id}: {res.name}")
        if res.details:
            print(f"      Metrics: {res.details}")
        if res.error_message:
            print(f"      Error: {res.error_message}")

    level = framework.compute_validation_level()
    print(f"\nFinal Validation Level: {level}")


if __name__ == "__main__":
    main()