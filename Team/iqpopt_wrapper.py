"""
iqpopt_wrapper.py  --  sponsor's IQPopt as its own model in the Team framework format.

    import iqpopt_wrapper as iqw
    ef.evaluate_model(iqw.iqpopt_function, settings, "IQPopt", "Qubit")

same input and output as qcbm_wrapper.py: training_data (already the training split), num_epochs, seed, num_samples in,
torch.long tensor (num_samples, n_bits) of 0/1 out.
up to 20 bits (the 12-bit qubit set) = 1 IQP circuit on all the bits. wider data = the overlapping windows in iqp_windows.py.
num_epochs = iqpopt training steps (per window on wide data).
"""

from __future__ import annotations

import numpy as np
import torch

import iqp_windows as iw
from qcbm_wrapper import to_bits

last_history = None # loss of every training step of the last call (mean over the windows), Main.py plots it


def iqpopt_function(training_data=None, seed: int = 42, num_epochs: int = 300, num_samples: int | None = None,
                    window: int = 20, overlap: int = 8, max_seconds: float = 0.0, verbose: bool = True, **settings) -> torch.Tensor:
    global last_history
    x = to_bits(training_data)
    n_rows, n_bits = x.shape
    wm = iw.fit_windows(x.numpy(), width=window, overlap=overlap, iters=int(num_epochs), seed=seed,
                        max_seconds=max_seconds, verbose=verbose)
    steps = min(len(w["losses"]) for w in wm["windows"]) # a time cap can stop the windows at different steps
    last_history = np.mean([w["losses"][:steps] for w in wm["windows"]], axis=0).tolist()
    print(f"IQPopt: {n_rows} training rows, {n_bits} bits, {len(wm['starts'])} window(s) of {wm['width']}, "
          f"trained in {wm['train_seconds']:.1f}s")
    return iw.sample_windows(wm, n_rows if num_samples is None else int(num_samples), seed).long()
