from dataclasses import dataclass, field
from typing import Any, Literal

TestStatus = Literal["PASSED", "FAILED", "NOT_DETECTED"]
ModelValidationLevel = Literal[
    "MATHEMATICALLY_VALIDATED",
    "DYNAMICALLY_VALIDATED",
    "STOCHASTICALLY_VALIDATED",
    "EMPIRICALLY_VALIDATED",
]


@dataclass
class TestResult:
    """Результат выполнения одиночного валидационного сценария."""
    test_id: str
    name: str
    status: TestStatus
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None