import argparse
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
from acmf.model.forcing import ForcingProfile
from acmf.decision.causal_mapping import PolicyIntervention
from acmf.scenarios.manager import DynamicScenario, ScenarioManager
from acmf.export.json_exporter import SimulationJSONExporter
from acmf.visualization.plots import SimulationVisualizer


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
    parser = argparse.ArgumentParser(description="ACMF 4.9.3.1 Scenario Runner CLI")
    parser.add_argument("--scenario", choices=["baseline", "shock", "policy"], default="shock", help="Тип сценария")
    parser.add_argument("--t_max", type=float, default=40.0, help="Время симуляции")
    parser.add_argument("--dt", type=float, default=0.05, help="Шаг интегрирования")
    parser.add_argument("--export_json", type=str, default="web/data/latest_simulation.json", help="Путь экспорта JSON")
    parser.add_argument("--save_plots", action="store_true", help="Сохранить графики PNG")
    args = parser.parse_args()

    params = build_default_parameters()
    mgr = ScenarioManager(params)

    # Определение сценариев
    policies = []
    jump_fn = None

    if args.scenario == "shock":
        # Импульсный внешний шок на шаге t in [5, 6]
        def jump_fn(x, t):
            if 5.0 <= t <= 5.2:
                return np.array([1.2, 0.8, 0.5])
            return np.zeros(3)

    elif args.scenario == "policy":
        # Шок в t=5 и регуляторная интервенция в t=8
        def jump_fn(x, t):
            if 5.0 <= t <= 5.2:
                return np.array([1.5, 1.0, 0.8])
            return np.zeros(3)

        policies = [
            (8.0, PolicyIntervention(u_reform=0.8, u_capacity=0.4, u_mitigation=0.5))
        ]

    scenario = DynamicScenario(
        name=args.scenario,
        t_span=(0.0, args.t_max),
        dt=args.dt,
        policy_schedule=policies,
        custom_jump_generator=jump_fn,
        random_seed=42,
    )

    print(f"Запуск сценария: {scenario.name} (T = {args.t_max}, dt = {args.dt})...")
    traj = mgr.run_scenario(scenario)

    # Экспорт данных в JSON
    json_path = SimulationJSONExporter.export_trajectory(
        trajectory=traj,
        metadata={"scenario": args.scenario, "t_max": args.t_max, "dt": args.dt},
        output_path=args.export_json,
    )
    print(f"[✓] Данные сохранены в JSON: {json_path}")

    # Построение графиков
    if args.save_plots:
        viz = SimulationVisualizer()
        viz.plot_time_series(traj, save_path="web/data/time_series.png")
        viz.plot_phase_portrait_2d(traj, save_path="web/data/phase_portrait.png")
        print("[✓] Графики сохранены в директории web/data/")


if __name__ == "__main__":
    main()