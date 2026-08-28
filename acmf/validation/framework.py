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

        # Уровень 1: Математическая корректность (TEST 00, 01, 16, 21)
        math_tests = ["TEST_00", "TEST_01", "TEST_16", "TEST_21"]
        if not all(statuses.get(t) == "PASSED" for t in math_tests):
            raise RuntimeError("Система не прошла базовую математическую валидацию.")

        # Уровень 2: Динамическая валидация (TEST 02, 03, 06, 07, 17, 18)
        dyn_tests = ["TEST_02", "TEST_03", "TEST_06", "TEST_07", "TEST_17", "TEST_18"]
        if not all(statuses.get(t) in ["PASSED", "NOT_DETECTED"] for t in dyn_tests):
            return "MATHEMATICALLY_VALIDATED"

        # Уровень 3: Стохастическая валидация (TEST 08, 11, 12, 13, 14, 15, 19)
        stoch_tests = ["TEST_08", "TEST_11", "TEST_12", "TEST_13", "TEST_14", "TEST_15", "TEST_19"]
        if not all(statuses.get(t) == "PASSED" for t in stoch_tests):
            return "DYNAMICALLY_VALIDATED"

        return "STOCHASTICALLY_VALIDATED"