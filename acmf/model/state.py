import numpy as np
from acmf.model.parameters import ModelParameters


class StateVector:
    """
    Вектор состояния ACMF с динамической размерностью:
    DIM = 2 * N_sub + 7
    """

    __slots__ = ("_data", "_N_sub", "_dim")

    def __init__(self, data: np.ndarray | None = None, N_sub: int | None = None) -> None:
        if data is None:
            raise ValueError("StateVector requires data array. Use np.zeros(2*N_sub+7) to construct.")
        
        data_arr = np.asarray(data, dtype=np.float64)
        
        if N_sub is None:
            # Вывод N_sub из длины массива: dim = 2*N_sub + 7
            if (len(data_arr) - 7) % 2 != 0:
                raise ValueError(f"Cannot infer N_sub from length {len(data_arr)}: (len-7) must be even.")
            self._N_sub = (len(data_arr) - 7) // 2
        else:
            self._N_sub = int(N_sub)
        
        self._dim = 2 * self._N_sub + 7
        
        if data_arr.shape != (self._dim,):
            raise ValueError(
                f"Expected shape ({self._dim},) for N_sub={self._N_sub}, got {data_arr.shape}"
            )
        self._data = data_arr

    @property
    def raw(self) -> np.ndarray:
        return self._data

    @property
    def N_sub(self) -> int:
        return self._N_sub

    @property
    def DIM(self) -> int:
        return self._dim

    @property
    def sid(self) -> np.ndarray:
        return self._data[0 : self._N_sub]

    @property
    def inst(self) -> float:
        return float(self._data[self._N_sub])

    @property
    def ch(self) -> float:
        return float(self._data[self._N_sub + 1])

    @property
    def prod(self) -> float:
        return float(self._data[self._N_sub + 2])

    @property
    def m(self) -> float:
        return float(self._data[self._N_sub + 3])

    @property
    def f(self) -> float:
        return float(self._data[self._N_sub + 4])

    @property
    def scar(self) -> float:
        return float(self._data[self._N_sub + 5])

    @property
    def ed(self) -> np.ndarray:
        start = self._N_sub + 6
        end = start + self._N_sub
        return self._data[start : end]

    @property
    def rec_debt(self) -> float:
        return float(self._data[2 * self._N_sub + 6])

    def is_in_domain(self, params: ModelParameters, tol: float = 1e-7) -> bool:
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
        if not self.is_in_domain(params):
            raise ValueError(f"Состояние нарушает границы домена Omega:\n{self._data}")

    def __repr__(self) -> str:
        return (
            f"StateVector(N_sub={self._N_sub}, DIM={self._dim},\n"
            f"  SID={self.sid},\n"
            f"  Inst={self.inst:.4f}, Ch={self.ch:.4f}, Prod={self.prod:.4f}, M={self.m:.4f},\n"
            f"  F={self.f:.4f}, Scar={self.scar:.4f},\n"
            f"  ED={self.ed}, RecDebt={self.rec_debt:.4f}\n"
            f")"
        )