"""
qcbm_wrapper.py  --  Khashim's QCBM (Quantum Circuit Born Machine) in the Team framework format.

Main.py runs it the same way as the other models:

    import qcbm_wrapper as qw
    ef.evaluate_model(qw.qcbm_function, settings)

evaluation_framework.py reads the CSV, keeps `epsilon` of the rows for training, and then calls
qcbm_function(**settings) with:
    training_data   torch.long tensor (n_train, n_bits) of 0/1 -- already the training split
                    (a csv path w/ a "bitstring" column, a list of "0101" strings or a list of lists also work)
    num_epochs      number of training steps (Adam, cosine lr, early stopping, best params kept)
    seed            random seed (same seed -> same output)
    epsilon         training fraction; the framework has already applied it, so it is not used here
    num_samples     how many bitstrings to generate

It returns a torch.long tensor (num_samples, n_bits) of 0/1, the same format as the VAE and transformer wrappers.
Bit order: the first bit of a row is qubit 0 (the most significant bit).
Limit: the QCBM simulates the full quantum state (2^n numbers), so one circuit only handles up to 18 bits.
Wider data (the 100/128-bit datasets) goes to iqp_windows.py instead: overlapping 20-bit windows, 1 iqpopt IQP circuit each,
sampled window by window (exact given the shared bits). Same input, same output, nothing to switch on.

My own scripts can still call it the old way: qcbm_function(trainingData="data/bas.csv", steps=400).

changes from sprint 9 (Nov 4):
    - starts from the Team copy (has hederik's fix: show_output is dropped before Config), saved with LF endings again
    - wide data no longer raises a ValueError, it runs iqp_windows.py (the 12-bit qubit set still uses the QCBM)
    - max_qubits 16 -> 18, the loss has no 2^n x 2^n kernel matrix anymore (speed_check.py: 18 bits = 1.2 GB, ~1 s/step)
    - logs the number of training rows and the training time on every call
    - new max_seconds (whole run, 0 = no cap), window and overlap
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

"""
the framework calls every model the same way: generated = MODEL(training_data), then MMD on both
trainingData = csv path w/ a "bitstring" column, list of "0101" strings, list of lists of 0/1 (hederik's framework), numpy array or torch tensor
returns torch.long (n_samples, n_bits) of 0/1, 1 bitstring per input row by default bc the ignite MMD needs the same size
trains on every row it gets, splitting train/test is the framework's job
first char of a bitstring = qubit 0 = most significant bit (same as data_generator.py)
only up to 18 bits per circuit, the qcbm simulates the full state (2^n numbers), the 100/128 bit datasets go to iqp_windows.py
need metrics vs the true dist. too? use qcbm.train()
"""

#framework runs from another folder, so add this one to the path before importing qcbm
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qcbm import Config, fit, sample_bits


# LOADING
def read_bitstring_csv(path) -> list:
    #read as strings, not ints, so leading zeros stay
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "bitstring" not in reader.fieldnames:
            raise ValueError(f"{path} needs a 'bitstring' column (got {reader.fieldnames})")
        return [r["bitstring"] for r in reader]


def to_bits(trainingData) -> torch.Tensor:
    # any of the input formats -> float64 (N, n_bits) tensor of 0/1
    if isinstance(trainingData, (str, os.PathLike)):
        rows = read_bitstring_csv(os.fspath(trainingData))
    elif isinstance(trainingData, (torch.Tensor, np.ndarray)):
        if trainingData.ndim != 2:
            raise ValueError(f"expected a 2D array (rows x bits), got shape {tuple(trainingData.shape)}")
        rows = trainingData.tolist()
    else:
        rows = list(trainingData)
    if len(rows) == 0:
        raise ValueError("trainingData is empty")

    try:
        rows = [list(r.strip()) if isinstance(r, str) else list(r) for r in rows]
    except TypeError:
        raise ValueError("each row must be a bitstring like '0101' or a list of 0/1 ints") from None
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        raise ValueError(f"all bitstrings must be the same length, got lengths {sorted(lengths)}")

    bit = {"0": 0, "1": 1, 0: 0, 1: 1} # 0.0 / 1.0 / True / False work too, same hash as 0 / 1
    try:
        return torch.tensor([[bit[v] for v in r] for r in rows], dtype=torch.float64)
    except (KeyError, TypeError):
        bad = next(v for r in rows for v in r if not isinstance(v, (str, int, float, np.generic)) or v not in bit)
        raise ValueError(f"bitstrings can only contain 0 and 1, found {bad!r}") from None


# WRAPPER
def qcbm_function(training_data=None, seed: int = 42, steps: int = 400, ansatz: str = "strong",
                  layers: int = 10, n_samples: int | None = None, max_qubits: int = 18, verbose: bool = False,
                  max_seconds: float = 0.0, window: int = 20, overlap: int = 8,
                  *, trainingData=None, num_epochs: int | None = None, num_samples: int | None = None,
                  epsilon: float | None = None, **settings) -> torch.Tensor:
    """Train a QCBM on the bitstrings and return new bitstrings sampled from it.

    In:  training_data (framework name) or trainingData (old name), plus num_epochs / num_samples / seed.
         epsilon is accepted and ignored, because the framework already split the data.
    Out: torch.long tensor (num_samples, n_bits) of 0/1. Default: 1 generated bitstring per input row.

        out = qcbm_function(training_data=train_tensor, num_epochs=100, seed=33, epsilon=0.3, num_samples=100)
        out.shape  # (100, n_bits)
    """
    if training_data is not None and trainingData is not None:
        raise ValueError("give the data as training_data OR trainingData, not both")
    data = training_data if training_data is not None else trainingData
    if data is None:
        raise ValueError("qcbm_function needs the data: pass training_data=... (or trainingData=...)")

    steps = steps if num_epochs is None else int(num_epochs)
    n_samples = n_samples if num_samples is None else int(num_samples)

    x = to_bits(data)
    n_rows, n_bits = x.shape

    settings = settings.copy()
    settings.pop("show_output", None)

    #128 bits would need 2^128 numbers in 1 circuit, so cut it into windows (steps = iqpopt steps per window)
    if n_bits > max_qubits:
        import iqp_windows as iw
        wm = iw.fit_windows(x.numpy(), width=window, overlap=overlap, iters=steps, seed=seed,
                            max_seconds=max_seconds, verbose=verbose)
        print(f"IQP windows: {n_rows} training rows, {n_bits} bits, {len(wm['starts'])} windows of {wm['width']}, "
              f"trained in {wm['train_seconds']:.1f}s")
        return iw.sample_windows(wm, n_rows if n_samples is None else n_samples, seed).long()

    cfg = Config(seed=seed, steps=steps, ansatz=ansatz, layers=layers, max_seconds=max_seconds, **settings)
    model = fit(x, cfg, verbose=verbose)
    print(f"QCBM: {n_rows} training rows, {n_bits} bits, {len(model.history)} epochs, trained in {model.train_seconds:.1f}s")
    with torch.no_grad():
        p = model()
    out = sample_bits(p, n_rows if n_samples is None else n_samples, torch.Generator().manual_seed(seed))
    return out.long()
