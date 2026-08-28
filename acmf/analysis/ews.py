from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EWSMetrics:
    variance: float
    ar1: float
    skewness: float
    recovery_time: float
    sample_size: int


@dataclass(frozen=True)
class EWSResult:
    metrics: EWSMetrics
    projection: np.ndarray
    risk_score: float
    risk_level: str


class EWSAnalyzer:
    def __init__(
        self,
        *,
        window_size: int,
        min_samples: int,
        risk_variance_threshold: float,
        risk_ar1_threshold: float,
        risk_skewness_threshold: float,
        risk_recovery_time_threshold: float,
    ) -> None:
        if window_size <= 1:
            raise ValueError(
                "window_size должен быть > 1"
            )

        if min_samples <= 1:
            raise ValueError(
                "min_samples должен быть > 1"
            )

        thresholds = (
            risk_variance_threshold,
            risk_ar1_threshold,
            risk_skewness_threshold,
            risk_recovery_time_threshold,
        )

        if not all(
            np.isfinite(value) and value >= 0.0
            for value in thresholds
        ):
            raise ValueError(
                "Пороговые значения EWS должны быть "
                "конечными и >= 0"
            )

        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.risk_variance_threshold = float(
            risk_variance_threshold
        )
        self.risk_ar1_threshold = float(
            risk_ar1_threshold
        )
        self.risk_skewness_threshold = float(
            risk_skewness_threshold
        )
        self.risk_recovery_time_threshold = float(
            risk_recovery_time_threshold
        )

    @staticmethod
    def _validate_projection(
        projection: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(
            projection,
            dtype=np.float64,
        )

        if value.ndim != 1:
            raise ValueError(
                "projection должен быть одномерным"
            )

        if value.size <= 1:
            raise ValueError(
                "projection содержит недостаточно наблюдений"
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                "projection содержит NaN или Inf"
            )

        return value

    def project(
        self,
        states: np.ndarray,
        left_vector: np.ndarray,
        equilibrium: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(
            states,
            dtype=np.float64,
        )

        w = np.asarray(
            left_vector,
            dtype=np.complex128,
        )

        x_eq = np.asarray(
            equilibrium,
            dtype=np.float64,
        )

        if x.ndim != 2:
            raise ValueError(
                "states должен быть двумерным"
            )

        if x.shape[1] != x_eq.size:
            raise ValueError(
                "equilibrium имеет неверную размерность"
            )

        if w.shape != x_eq.shape:
            raise ValueError(
                "left_vector имеет неверную размерность"
            )

        centered = x - x_eq

        projected = centered @ w

        return np.real(
            projected
        ).astype(
            np.float64,
            copy=False,
        )

    @staticmethod
    def variance(
        values: np.ndarray,
    ) -> float:
        if values.size < 2:
            return float("nan")

        return float(
            np.var(
                values,
                ddof=1,
            )
        )

    @staticmethod
    def ar1(
        values: np.ndarray,
    ) -> float:
        if values.size < 3:
            return float("nan")

        x = values[:-1]
        y = values[1:]

        x_centered = (
            x - np.mean(x)
        )
        y_centered = (
            y - np.mean(y)
        )

        denominator = float(
            np.dot(
                x_centered,
                x_centered,
            )
        )

        if denominator <= np.finfo(
            np.float64
        ).eps:
            return 0.0

        return float(
            np.dot(
                x_centered,
                y_centered,
            )
            / denominator
        )

    @staticmethod
    def skewness(
        values: np.ndarray,
    ) -> float:
        if values.size < 3:
            return float("nan")

        centered = (
            values
            - np.mean(values)
        )

        standard_deviation = float(
            np.std(
                values,
                ddof=1,
            )
        )

        if standard_deviation <= np.finfo(
            np.float64
        ).eps:
            return 0.0

        return float(
            np.mean(
                centered ** 3
            )
            / standard_deviation ** 3
        )

    @staticmethod
    def recovery_time(
        values: np.ndarray,
        times: np.ndarray,
        equilibrium: float,
        tolerance: float,
    ) -> float:
        if values.size != times.size:
            raise ValueError(
                "values и times должны иметь одинаковый размер"
            )

        if values.size < 2:
            return float("nan")

        if (
            not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "tolerance должен быть >= 0"
            )

        deviations = np.abs(
            values - equilibrium
        )

        outside = np.flatnonzero(
            deviations > tolerance
        )

        if outside.size == 0:
            return 0.0

        last_outside = int(
            outside[-1]
        )

        if last_outside >= values.size - 1:
            return float("inf")

        return float(
            times[last_outside + 1]
            - times[last_outside]
        )

    def compute(
        self,
        projection: np.ndarray,
        times: np.ndarray,
        recovery_tolerance: float,
    ) -> EWSResult:
        values = self._validate_projection(
            projection
        )

        time_values = np.asarray(
            times,
            dtype=np.float64,
        )

        if time_values.ndim != 1:
            raise ValueError(
                "times должен быть одномерным"
            )

        if time_values.size != values.size:
            raise ValueError(
                "times и projection должны иметь одинаковый размер"
            )

        if not np.all(np.isfinite(time_values)):
            raise ValueError(
                "times содержит NaN или Inf"
            )

        if np.any(
            np.diff(time_values) <= 0.0
        ):
            raise ValueError(
                "times должен быть строго возрастающим"
            )

        if values.size < self.min_samples:
            raise ValueError(
                "Недостаточно наблюдений для EWS"
            )

        window = values[
            -self.window_size:
        ]

        variance = self.variance(
            window
        )

        ar1 = self.ar1(
            window
        )

        skewness = self.skewness(
            window
        )

        recovery = self.recovery_time(
            values=window,
            times=time_values[
                -window.size:
            ],
            equilibrium=0.0,
            tolerance=recovery_tolerance,
        )

        metrics = EWSMetrics(
            variance=variance,
            ar1=ar1,
            skewness=skewness,
            recovery_time=recovery,
            sample_size=window.size,
        )

        variance_signal = (
            variance
            / max(
                self.risk_variance_threshold,
                np.finfo(
                    np.float64
                ).eps,
            )
        )

        ar1_signal = (
            max(ar1, 0.0)
            / max(
                self.risk_ar1_threshold,
                np.finfo(
                    np.float64
                ).eps,
            )
        )

        skewness_signal = (
            abs(skewness)
            / max(
                self.risk_skewness_threshold,
                np.finfo(
                    np.float64
                ).eps,
            )
        )

        recovery_signal = (
            recovery
            / max(
                self.risk_recovery_time_threshold,
                np.finfo(
                    np.float64
                ).eps,
            )
            if np.isfinite(recovery)
            else float("inf")
        )

        signals = np.array(
            [
                variance_signal,
                ar1_signal,
                skewness_signal,
                recovery_signal,
            ],
            dtype=np.float64,
        )

        risk_score = float(
            np.mean(
                np.minimum(
                    signals,
                    1.0,
                )
            )
        )

        risk_level = (
            "CRITICAL_TRANSITION_RISK_HIGH"
            if risk_score >= 1.0
            else "NORMAL"
        )

        return EWSResult(
            metrics=metrics,
            projection=values.copy(),
            risk_score=risk_score,
            risk_level=risk_level,
        )