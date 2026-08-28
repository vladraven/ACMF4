import numpy as np

from acmf.analysis.delay_spectrum import DelaySpectrumSolver
from acmf.analysis.equilibria import EquilibriumEngine
from acmf.analysis.jacobian import compute_dde_jacobians
from acmf.model.dynamics import compute_full_drift_vector
from acmf.model.forcing import ForcingProfile
from acmf.model.lag_estimation import estimate_lagged_derivatives
from acmf.model.parameters import ModelParameters
from acmf.model.state import StateVector
from acmf.validation.result import TestResult


def run_test_05(params: ModelParameters) -> TestResult:
    """TEST 05 — Branch-aware scan delay coupling and Hopf transitions."""

    forcing = ForcingProfile().evaluate(0.0)
    eq_engine = EquilibriumEngine()
    delay_solver = DelaySpectrumSolver()

    state_size = 9 + params.N_sub + 1

    def drift_det(x: np.ndarray) -> np.ndarray:
        st = StateVector(x)
        return compute_full_drift_vector(
            st,
            forcing,
            0.0,
            0.0,
            0.0,
            0.0,
            np.zeros(params.N_sub, dtype=np.float64),
            params,
        )

    def drift_full(
        x_curr: np.ndarray,
        x_delayed: np.ndarray,
        delay_value: float,
    ) -> np.ndarray:
        st = StateVector(x_curr)

        d_a_dt, d_prod_dt, d_inst_dt, d_agg_obs_dt = (
            estimate_lagged_derivatives(
                x_curr,
                x_delayed,
                delay_value,
                params,
            )
        )

        return compute_full_drift_vector(
            st,
            forcing,
            d_a_dt,
            d_prod_dt,
            d_inst_dt,
            d_agg_obs_dt,
            np.zeros(params.N_sub, dtype=np.float64),
            params,
        )

    healthy_seed = np.zeros(state_size, dtype=np.float64)
    healthy_seed[3:7] = 0.8
    healthy_seed[7] = 0.8 * params.F_max

    stressed_seed = np.zeros(state_size, dtype=np.float64)
    stressed_seed[0:params.N_sub] = 0.5 * params.SID_max
    stressed_seed[3:7] = 0.4
    stressed_seed[7] = 0.4 * params.F_max

    crisis_seed = np.zeros(state_size, dtype=np.float64)
    crisis_seed[0:params.N_sub] = min(
        params.SID_max - params.SID_buf,
        max(params.RefThresh, 0.8 * params.SID_max),
    )
    crisis_seed[3:7] = 0.1
    crisis_seed[7] = 0.1 * params.F_max

    threshold_seed = np.zeros(state_size, dtype=np.float64)
    threshold_seed[0:params.N_sub] = params.RefThresh
    threshold_seed[3] = 0.5
    threshold_seed[4:7] = 0.5
    threshold_seed[7] = 0.5 * params.F_max

    equilibria = eq_engine.scan_multistability(
        drift_det,
        [
            healthy_seed,
            stressed_seed,
            crisis_seed,
            threshold_seed,
        ],
    )

    if not equilibria:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan",
            status="FAILED",
            details={
                "reason": "no_equilibrium_branches_found",
                "equilibria_scanned": 0,
            },
        )

    delay_max = max(
        5.0,
        float(params.Delta_t),
        float(params.Delta_ref),
    )

    delay_min = min(
        max(float(params.Delta_t), 1e-3),
        delay_max,
    )

    delay_values = np.linspace(
        delay_min,
        delay_max,
        40,
        dtype=np.float64,
    )

    delay_norm_threshold = 1e-12

    branches_with_nontrivial_delay = 0
    max_delay_jacobian_norm = 0.0
    hopf_points: list[dict[str, float | complex | int]] = []

    for branch_index, equilibrium in enumerate(equilibria):
        equilibrium_state = equilibrium.state

        previous_real_part = None
        previous_delay = None

        branch_has_nontrivial_delay = False
        branch_max_a1_norm = 0.0

        for delay_value in delay_values:
            def dde_drift(
                x_curr: np.ndarray,
                x_delayed: np.ndarray,
                current_delay: float = float(delay_value),
            ) -> np.ndarray:
                return drift_full(
                    x_curr,
                    x_delayed,
                    current_delay,
                )

            a_0, a_1 = compute_dde_jacobians(
                dde_drift,
                equilibrium_state,
            )

            delay_jacobian_norm = float(np.linalg.norm(a_1))

            branch_max_a1_norm = max(
                branch_max_a1_norm,
                delay_jacobian_norm,
            )

            max_delay_jacobian_norm = max(
                max_delay_jacobian_norm,
                delay_jacobian_norm,
            )

            if delay_jacobian_norm <= delay_norm_threshold:
                continue

            branch_has_nontrivial_delay = True

            spectrum = delay_solver.find_roots(
                a_0,
                a_1,
                float(delay_value),
            )

            critical_root = spectrum.critical_root
            current_real_part = float(np.real(critical_root))

            if (
                previous_real_part is not None
                and previous_delay is not None
                and (
                    (previous_real_part < 0.0 <= current_real_part)
                    or (
                        previous_real_part > 0.0
                        >= current_real_part
                    )
                )
            ):
                if abs(float(np.imag(critical_root))) > 1e-2:
                    delay_step = float(
                        delay_value - previous_delay
                    )

                    transversality = (
                        (current_real_part - previous_real_part)
                        / delay_step
                        if delay_step != 0.0
                        else 0.0
                    )

                    hopf_points.append(
                        {
                            "branch": branch_index,
                            "delay": float(delay_value),
                            "critical_eigenvalue": critical_root,
                            "transversality": float(
                                transversality
                            ),
                        }
                    )

            previous_real_part = current_real_part
            previous_delay = float(delay_value)

        if branch_has_nontrivial_delay:
            branches_with_nontrivial_delay += 1

    if branches_with_nontrivial_delay == 0:
        return TestResult(
            test_id="TEST_05",
            name="Hopf DDE Bifurcation Scan",
            status="FAILED",
            details={
                "reason": "delay_coupling_not_effective_on_any_branch",
                "equilibria_scanned": len(equilibria),
                "branches_with_nontrivial_delay": 0,
                "max_delay_jacobian_norm": (
                    max_delay_jacobian_norm
                ),
            },
        )

    status = "PASSED" if hopf_points else "NOT_DETECTED"

    return TestResult(
        test_id="TEST_05",
        name="Hopf DDE Bifurcation Scan",
        status=status,
        details={
            "equilibria_scanned": len(equilibria),
            "branches_with_nontrivial_delay": (
                branches_with_nontrivial_delay
            ),
            "max_delay_jacobian_norm": (
                max_delay_jacobian_norm
            ),
            "hopf_count": len(hopf_points),
            "delay_min": float(delay_values[0]),
            "delay_max": float(delay_values[-1]),
        },
    )