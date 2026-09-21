# Experiment Report Template

## 1. Research Question
State the question being tested and the measurable hypothesis.

## 2. Experimental Setup
- Dataset / synthetic generator:
- Train/validation/test protocol:
- Random seeds:
- Software versions:
- Hardware / simulator:
- Quantum backend and noise model (if applicable):

## 3. Baselines
Describe the strongest relevant classical and/or non-quantum baseline and why it is matched fairly.

## 4. Method
Document preprocessing, model architecture, hyperparameters, optimization procedure, and stopping criteria.

## 5. Metrics
Report task quality together with resource metrics such as runtime, trainable parameters, circuit depth, qubit count, shots, memory, or energy where relevant.

## 6. Ablations
At minimum, vary one scientifically meaningful factor such as model depth, representation, noise level, data size, retrieval depth, LoRA rank, or modality availability.

## 7. Results
Populate tables only from executed experiments. Never enter illustrative values as if they were measured results.

| Method | Primary metric | Secondary metric | Runtime | Notes |
|---|---:|---:|---:|---|
| Baseline | — | — | — | Not yet run |
| Proposed | — | — | — | Not yet run |

## 8. Statistical Reporting
For stochastic experiments, report multiple seeds and mean ± standard deviation or confidence intervals where appropriate.

## 9. Error / Robustness Analysis
Describe failure cases, perturbation sensitivity, uncertainty, calibration, and distribution shift where applicable.

## 10. Reproducibility
Record the exact command, configuration, dependency versions, seed, and commit SHA used to generate the results.

## 11. Limitations
State dataset limitations, compute constraints, simulator-to-hardware gaps, and other threats to validity.

## 12. Future Work
Identify experiments that would materially strengthen the research claim.
