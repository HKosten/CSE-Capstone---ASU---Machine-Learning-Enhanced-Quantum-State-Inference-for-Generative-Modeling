"""QCBM in the team's wrapper format (same as faisal's vae_wrapper.py + jacqui's transformer_wrapper.py)

from qcbm_wrapper import qcbm_function
generated = qcbm_function("data/bas.csv")
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

# the framework calls every model the same way: generated = MODEL(training_data), then MMD on both
# trainingData = csv path w/ a "bitstring" column, list of "0101" strings, list of lists of 0/1 (hederik's framework), numpy array or torch tensor
# returns torch.long (n_samples, n_bits) of 0/1, 1 bitstring per input row by default bc the ignite MMD needs the same size
# trains on every row it gets, splitting train/test is the framework's job
# first char of a bitstring = qubit 0 = most significant bit (same as data_generator.py)
# only up to ~16 bits, the qcbm simulates the full state (2^n numbers) so the 128 bit datasets get a ValueError
# need metrics vs the true dist. too? use qcbm.train()

#framework runs from another folder, so add this one to the path before importing qcbm
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qcbm import Config, fit, sample_bits


# ─────────────────────────────────────────────── Loading ───────────────────────────────────────────────
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


# ─────────────────────────────────────────────── Wrapper ───────────────────────────────────────────────
def qcbm_function(trainingData, seed: int = 42, steps: int = 400, ansatz: str = "strong", layers: int = 10,
                  n_samples: int | None = None, max_qubits: int = 16, verbose: bool = False,
                  **settings) -> torch.Tensor:
    # n_samples=None -> 1 generated bitstring per input row
    x = to_bits(trainingData)
    n_rows, n_bits = x.shape
    #check before building anything, 128 bits would need 2^128 numbers
    if n_bits > max_qubits:
        raise ValueError(
            f"The QCBM simulates the full quantum state (2^n numbers), so it can only handle about "
            f"{max_qubits} qubits. This dataset has {n_bits} bits. Use a smaller dataset, like 3x3 "
            f"(9-bit) bars and stripes.")

    cfg = Config(seed=seed, steps=steps, ansatz=ansatz, layers=layers, **settings)
    model = fit(x, cfg, verbose=verbose)
    with torch.no_grad():
        p = model()
    out = sample_bits(p, n_rows if n_samples is None else n_samples, torch.Generator().manual_seed(seed))
    return out.long()
