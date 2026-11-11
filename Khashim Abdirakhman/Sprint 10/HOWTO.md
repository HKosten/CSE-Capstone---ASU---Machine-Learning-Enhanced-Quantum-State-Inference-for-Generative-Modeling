# Sprint 10 how-to

Run everything from this folder (`Khashim Abdirakhman/Sprint 10/`). The Team framework needs to be run from inside `Team/`.

```
pip install -r requirements.txt
```

## What is new

| File | What it does |
|---|---|
| `mmd.py` | Gaussian kernel computed as a product over bits, so MMD no longer needs a 2^n x 2^n matrix. Also `median_sigma` and `iqpopt_sigmas` |
| `qcbm.py` | Uses that MMD in the loss. New `Config.max_seconds` (time cap) |
| `qcbm_wrapper.py` | Same function the framework calls. One circuit up to 18 bits, wider data goes to the windows below. Logs training rows and training time |
| `iqp_windows.py` | Wide data: overlapping 20-bit windows, one IQPopt IQP circuit each, sampled window by window given the shared bits |
| `speed_check.py` | QCBM time and memory at 12, 14, 16, 18 bits |
| `epoch_test.py` | Where the QCBM stops improving on the 12-bit qubit set |
| `run_experiments.py` | The 9 runs (6 qubit, 3 wide) |
| `rescore.py` | Rescores saved samples with other kernel widths, plus validity and correlation checks. No retraining |
| `score_vs_sets.py` | Scores my saved samples against the test set, the training set and the whole dataset |
| `consistency_check.py` | Calls all four models like the framework does and checks type, shape and 0/1 values |
| `full128.py` | One IQPopt circuit on all 128 bits of Sin, compared with the window model |

## Commands

```
python mmd.py
python speed_check.py
python epoch_test.py
python run_experiments.py --only qubit --epochs 2000
python run_experiments.py --only wide
python rescore.py
python score_vs_sets.py
python consistency_check.py
python full128.py --minutes 8
```

## Versions that work

Tested on Windows, Python 3.13.5: pennylane 0.45.1, pennylane-lightning 0.45.0, jax 0.11.2, jaxlib 0.11.2, torch 2.9.0, numpy 2.3.3, iqpopt 2024.7.0 (see `requirements.txt` for the full list).

## The jax and pennylane crash

With this pennylane and jax pair, `IqpSimulator.sample()` and `.probs()` fail with `ImportError: cannot import name 'concrete_or_error' from 'jax.core'` when the parameters are jax arrays. Passing numpy arrays works. `iqp_windows.py` always converts to numpy before calling them, so the pinned versions above run without a downgrade. I did not test an older jax. A fresh venv with `pip install -r requirements.txt` ran the 12-bit and the wide path once (Python 3.13.5, Windows).

## Limits

- IQPopt state-vector sampling is practical up to about 26 qubits on this machine (100 shots: 20 bits 0.7 s, 24 bits 9 s, 26 bits 54 s, 28 bits 207 s). That is why the windows are 20 bits.
- The wide-set runs are short (10 minute cap per run), so they are under-trained. They show the pipeline works, not how good the model can get.
- IQPopt `bitflip=True` is a classical model, not the quantum circuit. It is only an extra baseline.
