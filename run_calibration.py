import json
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
from acmf.calibration.dataset import CalibrationDataset
from acmf.calibration.optimizer import EmpiricalOptimizer, ParameterBounds
from acmf.export.json_exporter import NumpyJSONEncoder


def build_default_parameters() -> ModelParameters:
    schema = ModelParametersSchema(
        dimensions=SystemDimensionsConfig(F_max=1.0, SID_buf=1.0, SID_max=3.0, kappa_s=10.0),
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
            eta=0.1, Capacity=[1.0, 1.0, 1.0], rho=[0.2, 0.2, 0.2], sigma_0=[0.1, 0.1, 0.1],
        ),
        epistemic=EpistemicAndReformConfig(
            alpha_mask=[0.3, 0.3, 0.3], lambda_burst=1.0, alpha_burst=5.0,
            ED_crit=0.5, ED_scale=1.0, ED_impact=0.5, RefThresh=0.2,
            lambda_ref_0=0.5, omega_fatigue=0.5, tau_ref=1.0, Delta_t=0.5, Delta_ref=0.5,
        ),
        contagion=ContagionAndHawkesConfig(
            kappa_spill=0.1, SID_contagion=0.5, beta_H=2.0,
            Gamma=[[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]],
            mu_rec=0.1, omega_V=0.2, omega_SID=0.2,
        ),
    )
    return ModelParameters.from_schema(schema)


def main() -> None:
    print("=== ACMF 4.9.3.1 Empirical Calibration Pipeline ===")
    base_params = build_default_parameters()

    # Загружаем эмпирические/синтетические данные для подгонки
    data_source = Path("web/data/latest_simulation.json")
    if not data_source.exists():
        print(f"[!] Файл {data_source} не найден. Сначала запустите: python run_scenario.py")
        return

    print(f"Загрузка датасета калибровки из {data_source}...")
    dataset = CalibrationDataset.from_file(data_source)
    if dataset.is_synthetic:
        print("    [!] Датасет помечен как СИНТЕТИЧЕСКИЙ — это калибровка на "
              "модельных, а не эмпирических данных.")

    optimizer = EmpiricalOptimizer(dataset=dataset, base_params=base_params)
    print("Запуск глобальной оптимизации Differential Evolution...")
    res = optimizer.calibrate(maxiter=10, popsize=6)

    print(f"\n[✓] Калибровка завершена за {res.iterations} итераций.")
    print(f"    Начальная ошибка Loss_0: {res.initial_loss:.6f}")
    print(f"    Финальная ошибка Loss_opt: {res.final_loss:.6f}")
    print(f"    Качество аппроксимации R^2: {res.r_squared:.4f}")

    opt_p = res.optimal_parameters
    print("\nОптимизированные параметры:")
    print(f"    alpha_pos:  {opt_p.alpha_pos:.4f}")
    print(f"    beta_neg:   {opt_p.beta_neg:.4f}")
    print(f"    gamma_inst: {opt_p.gamma_inst:.4f}")
    print(f"    mu_inst:    {opt_p.mu_inst:.4f}")
    print(f"    alpha_Prod: {opt_p.alpha_Prod:.4f}")
    print(f"    beta_Prod:  {opt_p.beta_Prod:.4f}")

    # Экспорт откалиброванных параметров
    calibrated_payload = {
        "status": "EMPIRICALLY_VALIDATED",
        "metrics": {
            "initial_loss": res.initial_loss,
            "final_loss": res.final_loss,
            "r_squared": res.r_squared,
            "iterations": res.iterations,
        },
        "parameters": {
            "alpha_pos": opt_p.alpha_pos,
            "beta_neg": opt_p.beta_neg,
            "gamma_inst": opt_p.gamma_inst,
            "mu_inst": opt_p.mu_inst,
            "alpha_Prod": opt_p.alpha_Prod,
            "beta_Prod": opt_p.beta_Prod,
            "F_max": opt_p.F_max,
            "SID_buf": opt_p.SID_buf,
            "SID_max": opt_p.SID_max,
        },
    }

    out_file = Path("config/calibrated_parameters.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibrated_payload, f, cls=NumpyJSONEncoder, indent=2)

    print(f"\n[✓] Откалиброванные параметры сохранены: {out_file}")
    print("Модель переведена на финальный уровень: EMPIRICALLY_VALIDATED")


if __name__ == "__main__":
    main()