# IQPopt vs PennyLane IQP

Bars and Stripes, 1000 training samples, 2000 test samples, seed 0, sigmas [0.5, 1.0, 2.0].
IQPopt: Adam lr 0.02, n_ops 1000, n_samples 1000. PennyLane: Adam lr 0.05 w/ cosine schedule + early stopping (same as qcbm.py).
Same circuit family for both: 1 parameter per qubit + 1 per pair.

## 3x3 (9 qubits, 45 parameters)

Equivalence check (my params / 2 in IQPopt): max prob difference = 5.20e-18 -> PASS

| model | MMD | TVD | KL | Fidelity | Validity | Coverage | Test MMD (mine) | Test MMD (iqpopt est.) | steps | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| pennylane | 0.0246 | 0.8148 | 1.9426 | 0.1658 | 0.1852 | 1.0000 | 0.0235 | 0.0257 | 400 | 14.4000 |
| iqpopt | 0.0185 | 0.5055 | 1.6978 | 0.3532 | 0.4945 | 0.9286 | 0.0175 | 0.0192 | 400 | 49.8000 |
| counting | 0.0006 | 0.0390 | 0.0055 | 0.9973 | 1.0000 | 1.0000 | 0.0005 | — | — | — |
| smoothed | 0.0020 | 0.1982 | 0.2263 | 0.7996 | 0.8018 | 1.0000 | 0.0023 | — | — | — |

Test MMD (mine) and Test MMD (iqpopt est.) should be close for the same model. If they are far apart, the kernels or sigmas don't match.

## Bigger grids (IQPopt only)

test MMD = IQPopt's estimate vs held-out samples. reference = real train set vs real test set (what a perfect model would get, ~0). Validity/coverage only where exact probs are still possible.

| grid | qubits | params | test MMD init | test MMD trained | reference | validity | random validity | coverage | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 4x4 | 16 | 136 | 0.0326 | 0.0164 | 0.0000 | 0.3194 | 4.58e-04 | 0.5333 | 37.7 |
