# Research Methodology

## Common Protocol

1. Define the research question and hypothesis.
2. Establish a transparent baseline.
3. Keep train/validation/test semantics explicit.
4. Fit preprocessing only on training data.
5. Use deterministic seeds where supported.
6. Report primary and secondary metrics.
7. Run ablations and robustness studies.
8. Record runtime and computational resources.
9. Archive configurations and raw outputs.
10. State limitations and threats to validity.

## Quantum Reporting

For quantum experiments report qubits, circuit depth, trainable parameters, shots, simulator/backend, noise model and classical controls.

## Integrity

A repository containing code is not automatically reproducible. Dataset version, preprocessing, hyperparameters, software environment and execution conditions must also be recorded.
