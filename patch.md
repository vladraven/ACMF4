1. Ложный fallback спектра DDE

acmf/analysis/delay_spectrum.py

eigvals(A₀+A₁) больше не подставляется, если Nelder–Mead не нашёл корень. Instantaneous spectrum только при Δ = 0 — это тождественное упрощение, не fallback.

2. state_dim из контракта модели

acmf/solver/engine.py + EM/Milstein/reflector

dim = 2·N_sub + 7 через StateLayout. История и SID-шум не хардкодятся как 13/3. dA/dt пишется в буфер и читается каузально.

3. Порог Хопфа и солверы — конфигурация

acmf/analysis/bifurcation.py

ContinuationConfig.hopf_imag_threshold. Промах поиска корней разрывает цепочку пересечения, а не изобретает λ.

4. Допуски равновесий

acmf/analysis/equilibria.py

residual_tol и uniqueness_tol живут в EquilibriumConfig, не в теле scan_multistability.

5. Масштабированный шаг Якобиана

acmf/analysis/jacobian.py

eps_j = eps · max(1, |x_j|). Именованный DEFAULT_FINITE_DIFFERENCE_STEP = 1e-6.

6. Устойчивая сигмоида

acmf/model/epistemic.py

B_burst без cutoff 100. Циклы по params.N_sub.

7. N_sub из контракта

metabolism / diffusion / dynamics / jumps / contagion

Аллокации и циклы читают params.N_sub. Три карты износа остаются структурными формулами 4.9.3.1.

8. TEST_21 — порядок, не две точки

acmf/validation/test_21_solver_independence.py

Попарное неувеличение расхождения + наблюдаемый порядок p ≥ 0.4.

9. TEST_14 — не магическое 0.1

acmf/validation/test_14_convergence.py

95% CI полуширина vs домен Inst [0,1] и согласие половин выборки.

10. CalibrationDataset

acmf/calibration/dataset.py

Флаг is_synthetic. EmpiricalDataset — алиас. Синтетический фит больше не зовётся EMPIRICALLY_VALIDATED.