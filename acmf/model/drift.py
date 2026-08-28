from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from acmf.model.parameters import ModelParameters


@dataclass(frozen=True)
class DriftResult:
    sid: np.ndarray
    inst: float
    fertility: float
    ch: float
    prod: float
    mobility: float
    scar: float


class DriftCalculator:
    def __init__(self, parameters: ModelParameters) -> None:
        self.parameters = parameters

    def s_plus(self, x: float | np.ndarray) -> float | np.ndarray:
        x_array = np.asarray(x, dtype=np.float64)
        result = np.logaddexp(
            0.0,
            self.parameters.k * x_array,
        ) / self.parameters.k

        if np.ndim(x) == 0:
            return float(result)

        return result

    def s_minus(self, x: float | np.ndarray) -> float | np.ndarray:
        return -self.s_plus(-np.asarray(x, dtype=np.float64))

    def effective_regeneration(
        self,
        r0: float,
        scar: float,
    ) -> float:
        return r0 * (
            1.0
            - self.parameters.gamma_R * scar
        )

    def tsm(
        self,
        dA_dt_lagged: float,
        dProd_dt_lagged: float,
        dInst_dt_lagged: float,
    ) -> float:
        exponent = (
            self.parameters.theta_A
            * abs(dA_dt_lagged)
            + self.parameters.theta_P
            * abs(dProd_dt_lagged)
            + self.parameters.theta_I
            * abs(dInst_dt_lagged)
        )

        return float(
            1.0 - np.exp(-exponent)
        )

    def institutional(
        self,
        inst: float,
        ch: float,
        mobility: float,
        g: float,
        r_eff: float,
        reform_impulse: float,
        agg_sid_true: float,
    ) -> float:
        positive = (
            self.parameters.alpha_pos
            * (
                r_eff * ch
                + self.parameters.gamma_inst
                * mobility
                * g
            )
            + reform_impulse
        )

        negative = (
            self.parameters.mu_inst
            + self.parameters.beta_neg
            * self.s_plus(agg_sid_true)
        )

        return float(
            positive * (1.0 - inst)
            - negative * inst
        )

    def fertility(
        self,
        fertility: float,
        mobility: float,
        g: float,
        sid_demographic: float,
    ) -> float:
        positive = (
            self.parameters.alpha_F
            * mobility
            * g
            * (
                self.parameters.F_max
                - fertility
            )
        )

        negative = (
            self.parameters.beta_F
            * self.s_plus(sid_demographic)
            * fertility
        )

        return float(
            positive - negative
        )

    def adaptation(
        self,
        inst: float,
        prod: float,
        ch: float,
        tsm_value: float,
    ) -> float:
        positive = (
            self.parameters.alpha_Ch
            * inst
            * prod
            * (1.0 - ch)
        )

        negative = (
            self.parameters.mu_Ch
            + self.parameters.beta_Ch
            * tsm_value
        ) * ch

        return float(
            positive - negative
        )

    def productivity(
        self,
        prod: float,
        automation: float,
        ch: float,
        sid_economic: float,
    ) -> float:
        positive = (
            self.parameters.alpha_Prod
            * automation
            * ch
            * (1.0 - prod)
        )

        negative = (
            self.parameters.beta_Prod
            * self.s_plus(sid_economic)
            * prod
        )

        return float(
            positive - negative
        )

    def mobility(
        self,
        prod: float,
        inst: float,
        mobility_value: float,
    ) -> float:
        positive = (
            self.parameters.alpha_M
            * prod
            * inst
            * (1.0 - mobility_value)
        )

        negative = (
            self.parameters.mu_M
            * mobility_value
        )

        return float(
            positive - negative
        )

    def scar(
        self,
        scar: float,
        agg_sid_true: float,
    ) -> float:
        accumulation = (
            self.parameters.gamma_scar
            * self.s_plus(
                agg_sid_true
                - self.parameters.Threshold_scar
            )
            * (1.0 - scar)
        )

        recovery = (
            self.parameters.mu_scar
            * scar
        )

        return float(
            accumulation - recovery
        )

    def sid(
        self,
        sid: np.ndarray,
        delta: np.ndarray,
        r_eff: float,
        inst: float,
        spillover: np.ndarray,
    ) -> np.ndarray:
        sid_value = np.asarray(
            sid,
            dtype=np.float64,
        )

        delta_value = np.asarray(
            delta,
            dtype=np.float64,
        )

        spillover_value = np.asarray(
            spillover,
            dtype=np.float64,
        )

        positive = (
            self.s_plus(delta_value)
            * (
                self.parameters.SID_max
                - sid_value
            )
            / self.parameters.SID_max
        )

        negative = (
            self.s_minus(delta_value)
            * (
                self.parameters.SID_buf
                + sid_value
            )
            / self.parameters.SID_buf
        )

        regeneration = (
            self.parameters.rho
            * r_eff
            * inst
            * self.s_plus(sid_value)
        )

        return (
            positive
            + negative
            - regeneration
            + spillover_value
        )

    def compute(
        self,
        sid: np.ndarray,
        inst: float,
        fertility: float,
        ch: float,
        prod: float,
        mobility: float,
        scar: float,
        automation: float,
        vulnerability: float,
        agency: float,
        r0: float,
        reform_impulse: float,
        agg_sid_true: float,
        delta: np.ndarray,
        spillover: np.ndarray,
        dA_dt_lagged: float,
        dProd_dt_lagged: float,
        dInst_dt_lagged: float,
    ) -> DriftResult:
        tsm_value = self.tsm(
            dA_dt_lagged=dA_dt_lagged,
            dProd_dt_lagged=dProd_dt_lagged,
            dInst_dt_lagged=dInst_dt_lagged,
        )

        r_eff = self.effective_regeneration(
            r0=r0,
            scar=scar,
        )

        inst_drift = self.institutional(
            inst=inst,
            ch=ch,
            mobility=mobility,
            g=agency,
            r_eff=r_eff,
            reform_impulse=reform_impulse,
            agg_sid_true=agg_sid_true,
        )

        fertility_drift = self.fertility(
            fertility=fertility,
            mobility=mobility,
            g=agency,
            sid_demographic=sid[2],
        )

        ch_drift = self.adaptation(
            inst=inst,
            prod=prod,
            ch=ch,
            tsm_value=tsm_value,
        )

        prod_drift = self.productivity(
            prod=prod,
            automation=automation,
            ch=ch,
            sid_economic=sid[1],
        )

        mobility_drift = self.mobility(
            prod=prod,
            inst=inst,
            mobility_value=mobility,
        )

        scar_drift = self.scar(
            scar=scar,
            agg_sid_true=agg_sid_true,
        )

        sid_drift = self.sid(
            sid=sid,
            delta=delta,
            r_eff=r_eff,
            inst=inst,
            spillover=spillover,
        )

        return DriftResult(
            sid=np.asarray(
                sid_drift,
                dtype=np.float64,
            ),
            inst=inst_drift,
            fertility=fertility_drift,
            ch=ch_drift,
            prod=prod_drift,
            mobility=mobility_drift,
            scar=scar_drift,
        )