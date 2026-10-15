# IQPopt vs PennyLane IQP

Bars and Stripes, 1000 training samples, 500 test samples, seed 0, sigmas [0.5, 1.0, 2.0].
IQPopt: Adam lr 0.02, n_ops 200, n_samples 200. PennyLane: Adam lr 0.05 w/ cosine schedule + early stopping (same as qcbm.py).
Same circuit family for both: 1 parameter per qubit + 1 per pair.

## 3x3 (9 qubits, 45 parameters)

Equivalence check (my params / 2 in IQPopt): max prob difference = 4.34e-18 -> PASS

| model | MMD | TVD | KL | Fidelity | Validity | Coverage | Test MMD (mine) | Test MMD (iqpopt est.) | steps | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| pennylane | 0.0396 | 0.9723 | 4.2010 | 0.0213 | 0.0277 | 0.5714 | 0.0400 | 0.0331 | 30 | 1.2000 |
| iqpopt | 0.0366 | 0.9521 | 3.2793 | 0.0436 | 0.0479 | 0.8571 | 0.0355 | 0.0279 | 30 | 7.5000 |
| counting | 0.0006 | 0.0390 | 0.0055 | 0.9973 | 1.0000 | 1.0000 | 0.0003 | — | — | — |
| smoothed | 0.0020 | 0.1982 | 0.2263 | 0.7996 | 0.8018 | 1.0000 | 0.0040 | — | — | — |

Test MMD (mine) and Test MMD (iqpopt est.) should be close for the same model. If they are far apart, the kernels or sigmas don't match.

