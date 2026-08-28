import numpy as np
from acmf.model.parameters import ModelParameters


class StateVector:
    """
    13-компонентное представление вектора состояния системы ACMF в замкнутом домене Omega:
    X = [SID^1, SID^2, SID^3, Inst, Ch, Prod, M, F, Scar, ED^1, ED^2, ED^3, RecDebt]
    """
    __slots__ = ("_data",)

    IDX_SID = slice(0, 3)
    IDX_INST = 3
    IDX_CH = 4
    IDX_PROD = 5
    IDX_M = 6
    IDX_F = 7
    IDX_SCAR = 8
    IDX_ED = slice(9, 12)
    IDX_REC_DEBT = 12
    DIM = 13

    def __init__(self, data: np.ndarray | None = None) -> None:
        if data is None:
            self._data = np.zeros(self.DIM, dtype=np.float64)
        else:
            if data.shape != (self.DIM,):
                raise ValueError(f"Размерность массива должна быть ({self.DIM},), получено {data.shape}")
            self._data = np.asarray(data, dtype=np.float64)

    @property
    def raw(self) -> np.ndarray:
        return self._data

    @property
    def sid(self) -> np.ndarray:
        return self._data[self.IDX_SID]

    @property
    def inst(self) -> float:
        return float(self._data[self.IDX_INST])

    @property
    def ch(self) -> float:
        return float(self._data[self.IDX_CH])

    @property
    def prod(self) -> float:
        return float(self._data[self.IDX_PROD])

    @property
    def m(self) -> float:
        return float(self._data[self.IDX_M])

    @property
    def f(self) -> float:
        return float(self._data[self.IDX_F])

    @property
    def scar(self) -> float:
        return float(self._data[self.IDX_SCAR])

    @property
    def ed(self) -> np.ndarray:
        return self._data[self.IDX_ED]

    @property
    def rec_debt(self) -> float:
        return float(self._data[self.IDX_REC_DEBT])

    def is_in_domain(self, params: ModelParameters, tol: float = 1e-7) -> bool:
        """Строгая проверка принадлежности текущего состояния домену Omega."""
        sid = self.sid
        if np.any(sid < -params.SID_buf - tol) or np.any(sid > params.SID_max + tol):
            return False

        if not (-tol <= self.inst <= 1.0 + tol):
            return False
        if not (-tol <= self.ch <= 1.0 + tol):
            return False
        if not (-tol <= self.prod <= 1.0 + tol):
            return False
        if not (-tol <= self.m <= 1.0 + tol):
            return False
        if not (-tol <= self.scar <= 1.0 + tol):
            return False

        if not (-tol <= self.f <= params.F_max + tol):
            return False

        if np.any(self.ed < -tol):
            return False

        if self.rec_debt < -tol:
            return False

        return True

    def validate(self, params: ModelParameters) -> None:
        """Генерирует исключение, если состояние выходит за границы Omega."""
        if not self.is_in_domain(params):
            raise ValueError(f"Состояние нарушает границы домена Omega:\n{self._data}")

    def __repr__(self) -> str:
        return (
            f"StateVector(\n"
            f"  SID={self.sid},\n"
            f"  Inst={self.inst:.4f}, Ch={self.ch:.4f}, Prod={self.prod:.4f}, M={self.m:.4f},\n"
            f"  F={self.f:.4f}, Scar={self.scar:.4f},\n"
            f"  ED={self.ed}, RecDebt={self.rec_debt:.4f}\n"
            f")"
        )