# ACMF4 correction pack

Baseline: commit 2b344e4

This pack contains only files whose validation logic was corrected against the existing repository structure.

Changed:
- acmf/validation/test_04_saddle_node.py
- acmf/validation/test_05_hopf.py
- acmf/validation/test_15_sobol.py
- acmf/validation/test_18_hysteresis.py
- acmf/validation/test_19_recovery_distribution.py
- acmf/validation/test_20_counterfactual.py
- acmf/validation/test_21_solver_independence.py

No new ACMF module names were introduced.

Important:
- TEST 04 no longer passes vacuously.
- TEST 05 explicitly fails when the supplied DDE callback has zero delay Jacobian.
- TEST 15 evaluates the actual ACMF drift instead of an unrelated synthetic formula.
- TEST 18 compares two trajectories with identical macro-state and different Scar.
- TEST 19 uses the actual diffusion implementation.
- TEST 21 performs a dt refinement comparison instead of a single-step-size comparison.
