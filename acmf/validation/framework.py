from typing import Callable
from acmf.validation.result import TestResult, ModelValidationLevel


class ValidationFramework:
    """Оркестратор запуска тестов TEST 00–21 и расчета итогового валидационного статуса."""

    def __init__(self) -> None:
        self.tests: dict[str, Callable[[], TestResult]] = {}
        self.results: list[TestResult] = []

    def register_test(self, test_id: str, test_fn: Callable[[], TestResult]) -> None:
        self.tests[test_id] = test_fn

    def run_all(self) -> list[TestResult]:
        self.results.clear()
        for test_id in sorted(self.tests.keys()):
            try:
                res = self.tests[test_id]()
            except Exception as exc:
                res = TestResult(
                    test_id=test_id,
                    name=self.tests[test_id].__name__,
                    status="FAILED",
                    error_message=str(exc),
                )
            self.results.append(res)
        return self.results

    def compute_validation_level(self) -> ModelValidationLevel:
        """Определяет уровень математической замкнутости и верификации модели."""
        statuses = {r.test_id: r.status for r in self.results}

        # Уровень 1: Математическая корректность (TEST 00, 01, 16, 21, 22, 23)
        # TEST_22/23 добавлены для закрытия ранее незакрытых caveat'ов:
        # TEST_22 — сертификат inward-pointing drift на границах Omega
        # (аналитическая лемма + исчерпывающая численная проверка на всём
        # операционном конверте параметров/состояний, а не в одной точке).
        # TEST_23 — строгая (matched Brownian path) сходимость EM,
        # устраняющая ограничение достоверности TEST_21 (несогласованные
        # реализации шума при разных dt).
        math_tests = ["TEST_00", "TEST_01", "TEST_16", "TEST_21", "TEST_22", "TEST_23"]
        if not all(statuses.get(t) == "PASSED" for t in math_tests):
            raise RuntimeError("Система не прошла базовую математическую валидацию.")

        # Уровень 2: Динамическая валидация
        # ИСПРАВЛЕНО: раньше TEST_04, TEST_05, TEST_09, TEST_10, TEST_20
        # вообще не участвовали в расчёте уровня — итоговый статус был
        # слеп к их реальному провалу (в частности к TEST_05, который
        # до фикса lag_estimation гарантированно фейлился).
        # TEST_04/05 по духу §TEST04/§TEST05 документа допускают
        # NOT_DETECTED как легитимный исход (отсутствие бифуркации в
        # скане — не ошибка модели). TEST_09/10/20 такого допущения не
        # имеют и должны реально PASSED.
        dyn_tests_allow_not_detected = ["TEST_04", "TEST_05"]
        dyn_tests_require_pass = [
            "TEST_02", "TEST_03", "TEST_06", "TEST_07", "TEST_17", "TEST_18",
            "TEST_09", "TEST_10", "TEST_20",
        ]
        if not all(statuses.get(t) in ["PASSED", "NOT_DETECTED"] for t in dyn_tests_allow_not_detected):
            return "MATHEMATICALLY_VALIDATED"
        if not all(statuses.get(t) == "PASSED" for t in dyn_tests_require_pass):
            return "MATHEMATICALLY_VALIDATED"

        # Уровень 3: Стохастическая валидация (TEST 08, 11, 12, 13, 14, 15, 19)
        stoch_tests = ["TEST_08", "TEST_11", "TEST_12", "TEST_13", "TEST_14", "TEST_15", "TEST_19"]
        if not all(statuses.get(t) == "PASSED" for t in stoch_tests):
            return "DYNAMICALLY_VALIDATED"

        return "STOCHASTICALLY_VALIDATED"