from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from acmf.solver.engine import TrajectoryResult


class SimulationVisualizer:
    """Генератор статических графиков и фазовых портретов."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")

    def plot_time_series(
        self,
        trajectory: TrajectoryResult,
        save_path: str | Path | None = None,
    ) -> None:
        """Строит многопанельный временной ряд динамики 13 компонент системы."""
        t = trajectory.times
        states = trajectory.states

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 1. SID компоненты
        axs[0].plot(t, states[:, 0], label="SID 1 (Institutional)", color="#d9534f", lw=2)
        axs[0].plot(t, states[:, 1], label="SID 2 (Economic)", color="#f0ad4e", lw=2)
        axs[0].plot(t, states[:, 2], label="SID 3 (Demographic)", color="#5bc0de", lw=2)
        axs[0].axhline(0.0, color="gray", linestyle="--", alpha=0.6)
        axs[0].set_ylabel("System Deficits (SID)")
        axs[0].legend(loc="upper left")
        axs[0].set_title("ACMF System Dynamics: Trajectory Analysis")

        # 2. Институты и капитал
        axs[1].plot(t, states[:, 3], label="Inst (Institutions)", color="#0275d8", lw=2)
        axs[1].plot(t, states[:, 4], label="Ch (Cohesion)", color="#5cb85c", lw=1.5)
        axs[1].plot(t, states[:, 5], label="Prod (Production)", color="#6f42c1", lw=1.5)
        axs[1].plot(t, states[:, 6], label="M (Epistemic Matrix)", color="#20c997", lw=1.5)
        axs[1].plot(t, states[:, 7], label="F (Capital)", color="#17a2b8", linestyle="-.", lw=1.5)
        axs[1].set_ylabel("Capacities & Capital")
        axs[1].legend(loc="upper left")

        # 3. Шрам и долги
        axs[2].plot(t, states[:, 8], label="Scar (Hysteresis Memory)", color="#292b2c", lw=2.5)
        axs[2].plot(t, states[:, 12], label="RecDebt (Recovery Debt)", color="#e83e8c", linestyle="--", lw=1.5)
        axs[2].set_xlabel("Time (t)")
        axs[2].set_ylabel("Memory & Debts")
        axs[2].legend(loc="upper left")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_phase_portrait_2d(
        self,
        trajectory: TrajectoryResult,
        x_idx: int = 0,
        y_idx: int = 3,
        x_label: str = "SID 1 (Institutional Deficit)",
        y_label: str = "Inst (Institutional Capacity)",
        save_path: str | Path | None = None,
    ) -> None:
        """Строит 2D фазовый портрет эволюции системы."""
        x_data = trajectory.states[:, x_idx]
        y_data = trajectory.states[:, y_idx]

        plt.figure(figsize=(8, 6))
        plt.plot(x_data, y_data, color="#0275d8", lw=2, label="Trajectory")
        plt.scatter([x_data[0]], [y_data[0]], color="#5cb85c", s=100, zorder=5, label="Start")
        plt.scatter([x_data[-1]], [y_data[-1]], color="#d9534f", s=100, zorder=5, label="End")

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Phase Portrait: {y_label} vs {x_label}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_ews_signals(
        self,
        times: np.ndarray,
        z_series: np.ndarray,
        var_series: np.ndarray,
        ar1_series: np.ndarray,
        save_path: str | Path | None = None,
    ) -> None:
        """Строит опережающие индикаторы раннего предупреждения (EWS)."""
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(times, z_series, color="#292b2c", lw=1.5)
        axs[0].set_ylabel("Critical Mode Z(t)")
        axs[0].set_title("Early Warning Signals (EWS) Degradation Metrics")

        axs[1].plot(times, var_series, color="#d9534f", lw=2)
        axs[1].set_ylabel("Rolling Variance Var(Z)")

        axs[2].plot(times, ar1_series, color="#f0ad4e", lw=2)
        axs[2].set_ylabel("Autocorrelation AR(1)")
        axs[2].set_xlabel("Time (t)")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300)
            plt.close()
        else:
            plt.show()