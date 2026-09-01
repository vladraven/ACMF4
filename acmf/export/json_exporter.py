import json
from pathlib import Path
from typing import Any
import numpy as np
from acmf.solver.engine import TrajectoryResult
from acmf.validation.result import TestResult


class NumpyJSONEncoder(json.JSONEncoder):
    """Кастомный энкодер для сериализации numpy-структур и комплексных чисел."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (complex, np.complex128, np.complex64)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)


class SimulationJSONExporter:
    """Экспортер данных симуляции и валидации в JSON для веб-дашборда."""

    @staticmethod
    def export_trajectory(
        trajectory: TrajectoryResult,
        ews_data: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
        output_path: str | Path = "simulation_output.json",
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            # Every payload from this exporter comes from ACMFEngine.simulate,
            # never from an external/observational source — mark it explicitly
            # so CalibrationDataset.from_file cannot mistake it for real data.
            "is_synthetic": True,
            "metadata": metadata or {},
            "times": trajectory.times,
            "states": {
                "sid_1": trajectory.states[:, 0],
                "sid_2": trajectory.states[:, 1],
                "sid_3": trajectory.states[:, 2],
                "inst": trajectory.states[:, 3],
                "ch": trajectory.states[:, 4],
                "prod": trajectory.states[:, 5],
                "m": trajectory.states[:, 6],
                "f": trajectory.states[:, 7],
                "scar": trajectory.states[:, 8],
                "ed_1": trajectory.states[:, 9],
                "ed_2": trajectory.states[:, 10],
                "ed_3": trajectory.states[:, 11],
                "rec_debt": trajectory.states[:, 12],
            },
            "phase_space": {
                "sid_1_vs_inst": np.column_stack((trajectory.states[:, 0], trajectory.states[:, 3])),
                "sid_2_vs_inst": np.column_stack((trajectory.states[:, 1], trajectory.states[:, 3])),
                "sid_1_vs_sid_2": np.column_stack((trajectory.states[:, 0], trajectory.states[:, 1])),
                "f_vs_prod": np.column_stack((trajectory.states[:, 7], trajectory.states[:, 5])),
            },
            "diagnostics": {
                "reflections_count": sum(1 for d in trajectory.diagnostics if d.reflected),
                "overshoot_max": [float(np.max(d.raw_overshoot)) for d in trajectory.diagnostics] if trajectory.diagnostics else [],
            },
            "ews": ews_data or {},
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=NumpyJSONEncoder, indent=2)

        return out_file

    @staticmethod
    def export_validation_suite(
        results: list[TestResult],
        validation_level: str,
        output_path: str | Path = "validation_report.json",
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "validation_level": validation_level,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.status == "PASSED"),
            "failed_tests": sum(1 for r in results if r.status == "FAILED"),
            "results": [
                {
                    "test_id": r.test_id,
                    "name": r.name,
                    "status": r.status,
                    "details": r.details,
                    "error_message": r.error_message,
                }
                for r in results
            ],
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=NumpyJSONEncoder, indent=2)

        return out_file