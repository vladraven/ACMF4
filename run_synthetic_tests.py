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
from acmf.validation.synthetic_suite import SyntheticTestSuite
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
    print("=== ACMF 4.9.3.1 Synthetic Stress Suite ===")
    params = build_default_parameters()
    suite = SyntheticTestSuite(params)

    test_results = suite.run_all()
    output_payload = {
        "suite_name": "ACMF 4.9.3.1 Synthetic Stress Scenarios",
        "total_tests": len(test_results),
        "passed_count": sum(1 for t in test_results if t["passed"]),
        "tests": [],
    }

    for item in test_results:
        traj = item["trajectory"]
        status_flag = "[✓] PASS" if item["passed"] else "[✗] FAIL"
        print(f"{status_flag} | {item['id']}: {item['name']}")

        test_data = {
            "id": item["id"],
            "name": item["name"],
            "passed": item["passed"],
            "summary": item["summary"],
            "times": traj.times,
            "states": {
                "sid_1": traj.states[:, 0],
                "sid_2": traj.states[:, 1],
                "sid_3": traj.states[:, 2],
                "inst": traj.states[:, 3],
                "ch": traj.states[:, 4],
                "prod": traj.states[:, 5],
                "m": traj.states[:, 6],
                "f": traj.states[:, 7],
                "scar": traj.states[:, 8],
                "rec_debt": traj.states[:, 12],
            },
            "phase_space": {
                "sid_1_vs_inst": np.column_stack((traj.states[:, 0], traj.states[:, 3])),
                "sid_2_vs_inst": np.column_stack((traj.states[:, 1], traj.states[:, 3])),
                "sid_1_vs_sid_2": np.column_stack((traj.states[:, 0], traj.states[:, 1])),
                "f_vs_prod": np.column_stack((traj.states[:, 7], traj.states[:, 5])),
            },
            "ews": item["ews"],
        }
        output_payload["tests"].append(test_data)

    out_file = Path("web/data/synthetic_tests.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, cls=NumpyJSONEncoder, indent=2)

    print(f"\n[✓] Синтетический пакет экспортирован: {out_file}")


if __name__ == "__main__":
    main()