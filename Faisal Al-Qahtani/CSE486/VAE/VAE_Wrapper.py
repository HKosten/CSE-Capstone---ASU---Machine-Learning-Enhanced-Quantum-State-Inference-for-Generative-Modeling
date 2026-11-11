"""
vae_wrapper.py  --  the VAE in the team's wrapper format.

Every model is wrapped in a function so the shared evaluation framework can call all of them
the same way. This follows the format of the transformer wrapper (`transformer_function`):
ONE input, `trainingData`, and the function returns the bitstrings the model generates.

    from vae_wrapper import vae_function

    output = vae_function("bitstrings.csv")        # CSV with a "bitstring" column
    for r in output:
        print("".join(map(str, r.tolist())))       # one generated bitstring per line

What it does
  * trainingData: a path to a CSV with a "bitstring" column (like the transformer wrapper),
    or an array / tensor / list of "0101..." strings.
  * It trains on EVERY row it is given (train_frac=1.0). Splitting the data into train and
    test is the framework's job, so pass it the training part.
  * It returns only the generated bitstrings, as a torch.long tensor of shape
    (n_samples, n_bits) with values 0/1, the same dtype and layout as the transformer wrapper.
  * By default it returns as many bitstrings as it was given rows, like the transformer
    wrapper. This matters because the MMD code (ignite) needs both sets to be the same size.
    Pass n_samples=... to change that.
  * Everything else is an optional keyword argument with a default: seed, epochs, n_samples,
    and the model settings of run_vae() (latent_dim, hidden_dim, batch_size, lr, beta,
    train_frac).

Need the loss curves, the held-out data, or the settings used as well? Call run_vae() in
vae_interface.py instead. If the team's agreed format changes, change this file.
"""

from __future__ import annotations

import os

import torch

from vae_interface import load_bitstrings, run_vae


def vae_function(trainingData, seed: int = 50, epochs: int = 500, n_samples: int | None = None,
                 **settings) -> torch.Tensor:
    """Train the VAE on `trainingData` and return generated bitstrings as a torch.long tensor
    of shape (n_samples, n_bits). n_samples=None means one generated bitstring per input row."""
    if isinstance(trainingData, (str, os.PathLike)):
        trainingData = load_bitstrings(os.fspath(trainingData))
    if n_samples is None:
        n_samples = len(trainingData)
    settings.setdefault("train_frac", 1.0)  # the framework splits the data, not the wrapper
    out = run_vae(trainingData, seed=seed, epochs=epochs, n_samples=n_samples, **settings)
    return out["generated_samples"].long()
