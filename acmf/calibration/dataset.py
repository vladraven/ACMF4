from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json


@dataclass(frozen=True)
class EmpiricalDataset:
    """Контейнер эмпирических наблюдаемых временных рядов."""
    times: np.ndarray
    sid_obs: np.ndarray     # Форма (N, 3)
    inst_obs: np.ndarray    # Форма (N,)
    prod_obs: np.ndarray    # Форма (N,)
    weights: np.ndarray     # Веса компонент при расчете ошибки

    @classmethod
    def from_synthetic_or_file(
        cls,
        json_path: str | Path,
        weights: list[float] | None = None,
    ) -> "EmpiricalDataset":
        """Загружает наблюдаемые ряды из JSON-файла."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Поддержка структуры как прямого экспорта, так и синтетических тестов
        if "tests" in raw and len(raw["tests"]) > 0:
            data_block = raw["tests"][0]
        else:
            data_block = raw

        times = np.array(data_block["times"], dtype=np.float64)
        states = data_block["states"]

        sid = np.column_stack((
            np.array(states["sid_1"], dtype=np.float64),
            np.array(states["sid_2"], dtype=np.float64),
            np.array(states["sid_3"], dtype=np.float64),
        ))
        inst = np.array(states["inst"], dtype=np.float64)
        prod = np.array(states["prod"], dtype=np.float64)

        w = np.array(weights if weights else [1.0, 1.0, 1.0, 2.0, 1.5], dtype=np.float64)

        return cls(
            times=times,
            sid_obs=sid,
            inst_obs=inst,
            prod_obs=prod,
            weights=w,
        )