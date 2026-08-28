# ACMF 4.9.3.1 correction pack — b015125

This pack is based on the actual working commit b015125.

It:
- preserves the existing repository paths;
- removes the two accidental duplicate modules introduced after 2b344e4;
- removes model-parameter hardcode from run_validation.py;
- makes the final validation level account for the dynamic/decision tests that were previously ignored;
- preserves the honest TEST_05 failure rather than manufacturing a PASS;
- includes the full audit findings.

Run REMOVE_ACCIDENTAL_FILES.ps1 once after extraction.

No new ACMF model/analysis/solver module is introduced by this pack.
