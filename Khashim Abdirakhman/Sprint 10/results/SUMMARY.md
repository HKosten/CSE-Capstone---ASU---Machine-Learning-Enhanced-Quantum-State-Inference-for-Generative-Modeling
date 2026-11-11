# Sprint 10 summary

Nov 4. Everything here was run on one Windows laptop (8 cores, 15.8 GB RAM). Raw numbers are in `results/*.json` and `results/runs/`.

**Read this first:** the three wide-set runs (DNA, Sin, Bars+Stripes) are under-trained: 500 IQPopt steps per window, 39 parameters per window, 6 to 7 minutes per dataset, one seed. They only show that the pipeline works end to end (train windows, stitch, sample, score). They do not show how good the model can get, and on these runs it is no better than coin flips.

## 1. Product-kernel MMD (`mmd.py`)

The Gaussian kernel on 0/1 strings is a product over bits, so `K @ v` is n small 2x2 steps. Same numbers as the matrix version (max difference 7e-18 at 3, 8 and 12 bits; a 20-step QCBM fit gives the same best loss as the old code to 1e-16).

QCBM speed and memory (`speed_check.py`, 10 steps after a warm-up, 300 training rows):

| bits | seconds per step | peak RAM | old kernel matrix would need |
|---|---|---|---|
| 12 | 0.17 | 0.47 GB | 0.12 GB |
| 14 | 0.26 | 0.50 GB | 2 GB |
| 16 | 0.41 | 0.64 GB | 32 GB |
| 18 | 0.95 | 1.19 GB | 512 GB |

`max_qubits` in the wrapper is now 18.

## 2. Can IQPopt do the full width?

- Training: yes. Nearest-neighbour gates at 100 and 128 qubits, 5 steps in about 5 s including compile.
- State-vector sampling (100 shots, lightning.qubit): 20 bits 0.7 s, 24 bits 9 s, 26 bits 54 s, 28 bits 207 s. Largest practical width here is about 26. At 30 it would need more RAM than the machine has (computed, not run).
- `bitflip=True` samples at 100 and 128 bits in 5 to 7 s per 100 shots, but it is a classical model, so it is only an extra baseline.
- Crash: with pennylane 0.45.1 and jax 0.11.2, `sample` and `probs` fail on jax-array parameters. Numpy parameters work, and `iqp_windows.py` always uses them.

So the wide model is 20-bit windows overlapping by 8 (8 windows at 100 bits, 10 at 128), one IQPopt circuit each, stitched with exact conditional sampling. Stitching check: on a sticky Markov chain (truth 0.95 agreement between neighbours) overlap 8 gives 0.947 to 0.951 across window seams, overlap 0 gives 0.505.

## 3. Epoch test (12-bit qubit set, epsilon 0.5, seed 33)

Stopped at epoch 3419 (no improvement for 1000 epochs, 822 s). Best epoch 2419. Train MMD at epochs 1000, 2000, 3000: 0.1503, 0.1497, 0.1497. It flattens early and at a high value. All qubit runs below use 2419 epochs.

## 4. The 9 runs (score = framework ignite MMD, lower is closer)

12-bit qubit set, QCBM, cap 10 min per run:

| epsilon | seed 33 | seed 34 | note |
|---|---|---|---|
| 0.1 | 0.4389 | 0.4413 | both hit the cap (1456 and 1743 epochs) |
| 0.5 | 0.4315 | 0.4334 | 2419 epochs |
| 0.9 | 0.4329 | 0.4364 | 2419 epochs |

For scale, coin flips score 0.558 here and the Transformer 0.050. The QCBM training problem is still open (see section 7).

Wide sets, epsilon 0.5, seed 33, cap 10 min (none hit it):

| dataset | IQP windows | bitflip baseline (500 steps, full width) | windows train time |
|---|---|---|---|
| DNA (100 bits) | 0.00000 | 0.00000 | 350 s |
| Sin (128) | 0.13115 | 0.13105 | 429 s |
| Bars+Stripes (128) | 0.00192 | 0.00192 | 380 s |

## 5. The MMD kernel problem (rescored saved samples, no retraining)

All sample sets scored against the same 500 test rows (epsilon 0.5, seed 33 split; no model trained on them). Numbers are sqrt of MMD, lower is closer. Full tables in `results/rescore.json`.

With the framework's ignite kernel (variance 1) the wide-data scores cannot rank models:

| Sin (128 bits) | framework | median heuristic (sigma 2.83) | valid rows |
|---|---|---|---|
| coin flips | 0.1311 | 0.3255 | 0% |
| LSTM | 0.1311 | 0.3258 | 0% |
| VAE | 0.1283 | 0.1776 | 0% |
| Transformer | 0.0510 | 0.0507 | 82% |
| IQP windows (sprint 10) | 0.1312 | 0.4789 | 0% |

- On DNA the framework gives 0.0000 to LSTM, VAE, coin flips and both Sprint 10 models. On Bars+Stripes it gives 0.00192 to all of them. Only the Transformer differs (0.053), and it is the worst number even though 83% of its rows are valid bars-and-stripes grids and its neighbour correlation (0.272) matches the data (0.249). The framework score is high for the Transformer because its output repeats rows, not because it is further from the data.
- With the median-heuristic width the Sin ranking matches the validity check: Transformer first, then VAE, then LSTM and coin flips tied. The iqpopt `sigma_heuristic` gives the same order.
- Bars+Stripes with the median width: Transformer 0.0564, VAE 0.0635, LSTM 0.0860, coin flips 0.0869. Same order, smaller gaps.
- DNA has almost no structure to find (neighbour correlation 0.09), so every model is close to coin flips under every kernel. That dataset may not be a useful test of anything.
- A kernel width that depends on the data is a one-line change, but I did not touch the Team framework.

## 6. Validity and correlation checks (500 generated rows each)

| | Sin valid rows | Sin neighbour corr (real 0.969) | B+S valid rows | B+S neighbour corr (real 0.249) |
|---|---|---|---|---|
| Transformer | 82.0% | 0.962 | 83.2% | 0.272 |
| VAE | 0.0% | 0.525 | 0.0% | 0.106 |
| LSTM | 0.0% | 0.443 | 0.0% | 0.147 |
| IQP windows | 0.0% | 0.410 | 0.0% | 0.097 |
| bitflip | 0.0% | 0.299 | 0.0% | 0.054 |
| coin flips | 0.0% | 0.033 | 0.0% | 0.033 |

On Sin the IQP windows are worse than coin flips on per-bit mean error (0.351 against 0.024) and on average pairwise distance (32.8 against 64.1 for the data). The windows learned some local correlation (0.410) and nothing else in 500 steps.

## 7. Follow-up checks (Nov 4, evening)

| Check | Result |
|---|---|
| Saved samples vs test, train and whole dataset (`score_vs_sets.py`, `results/score_vs_sets.json`) | Framework score, test / train / whole: Sin windows 0.1312 / 0.1292 / 0.1300, Bars+Stripes windows 0.00192 / 0.00206 / 0.00264, Qubit QCBM eps 0.5 seed 33 0.4315 / 0.4285 / 0.4384. Almost the same on all three, so nothing is memorised. Only the Qubit eps 0.1 seed 34 run is better on train (0.394) than test (0.441) |
| Output format against the other models (`consistency_check.py`, `results/consistency_check.json`) | Asked for 137 samples on the 12-bit and DNA sets: all four return (137, n_bits) with only 0/1 values. QCBM, Transformer and VAE return int64 tensors. LSTM returns a Python list, same shape and values. The 13 files saved by the last Main.py run are all (1000, n_bits) |
| Fresh venv (Python 3.13.5, `pip install -r requirements.txt`) | Install worked, iqpopt built from git as 2024.7.0. QCBM on the 12-bit set, 20 epochs, 3.4 s: int64 tensor (100, 12). Wide path on DNA, 3 steps per window, 9 s: int64 tensor (100, 100). My first smoke script had a path bug and failed, the rerun passed |
| One IQPopt circuit on all 128 bits of Sin (`full128.py`, `results/full128_sin.json`) | 255 parameters, 2500 steps in 486 s. IQPopt-estimated MMD squared: full circuit 0.0287 (std 0.0008 over 5 keys), untrained 0.262, window samples 0.0371, coin flips 0.0294. Correlation of bit 0 and bit 127: real data 0.980, window samples 0.064, full circuit -0.003 |

What the last row says:
- The full-width circuit is not better than coin flips (0.0287 against 0.0294). The windows are worse (0.0371). The full-circuit number is an IQPopt estimate, the others are sample estimates, so the small gap to coin flips should not be read as a win.
- Neither model reproduces the bit 0 to bit 127 link. In Sin bit 127 sits next to bit 0 in the repeating pattern, so the real correlation is 0.98. The windows are 88 bits apart there by design. The full circuit has only neighbour-pair gates along a line, so it cannot connect them either.
- Training a full-width circuit costs about 0.19 s per step at 128 bits. Sampling from it is the part that does not work.

## 8. Not done

- The 12-bit QCBM training problem (train MMD stuck at 0.15, per-bit means far from the data).
- Better wide-set models: more steps, all-pairs gates, several seeds. The wide results above are still only pipeline checks.
- The extra epsilon and seed runs on the wide sets (skipped on request).
- A kernel-width proposal for the team. The Team framework is unchanged.
- Review and commit. Nothing is committed. `Team/` has the new `qcbm.py`, `mmd.py`, `qcbm_wrapper.py`, `iqp_windows.py` and a docstring edit in `Main.py`.
- Checks I did not run: a jax downgrade, a fresh venv on another OS or Python version, the full Main.py with the new wrapper (about 16 minutes; only the 12-bit and DNA paths were run through the framework in a scratch folder).
- Left alone because they are not mine: the LSTM returning a list, and the root README still saying the QCBM skips wide data.
