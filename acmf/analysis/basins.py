from typing import Callable
import numpy as np


class BasinClassifier:
    """
    Классификация бассейнов притяжения на основе асимптотического поведения траекторий.
    """

    @staticmethod
    def classify_trajectory(
        trajectory_final_state: np.ndarray,
        healthy_target: np.ndarray,
        sid_max: float,
        tol: float = 0.2,
    ) -> str:
        """
        Классифицирует режим:
        - 'Healthy' (система вернулась в окрестность здорового режима)
        - 'Anarchy' (институциональный коллапс SID^1 -> SID_max)
        - 'Default' (экономический коллапс SID^2 -> SID_max)
        - 'Depopulation' (демографический коллапс SID^3 -> SID_max)
        - 'Hybrid' (множественный коллапс)
        """
        sid_final = trajectory_final_state[0:3]

        if np.linalg.norm(trajectory_final_state - healthy_target) < tol:
            return "Healthy"

        collapsed = sid_final >= (sid_max - tol)
        n_collapsed = int(np.sum(collapsed))

        if n_collapsed >= 2:
            return "Hybrid"
        if collapsed[0]:
            return "Anarchy"
        if collapsed[1]:
            return "Default"
        if collapsed[2]:
            return "Depopulation"

        return "Intermediate"