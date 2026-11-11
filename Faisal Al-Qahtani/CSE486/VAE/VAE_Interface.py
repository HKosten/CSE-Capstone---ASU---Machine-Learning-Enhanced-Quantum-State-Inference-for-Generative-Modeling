"""
vae_interface.py  --  the "model wrap": one function the team's evaluation framework calls.

The sprint task was: turn each person's model into a function that takes the data and
adjustable settings (seed, train/test split, epochs, ...) and returns what the shared
evaluation framework needs (generated samples + loss). That function is `run_vae`.

This file is deliberately separate from vae_model.py. If the team changes what the
framework expects (different argument names, extra outputs, ...), only THIS file changes.

Typical use from the framework:

    from vae_interface import load_bitstrings, run_vae

    data = load_bitstrings("bars_stripes_128.csv")          # (N, n_bits) float tensor of 0/1
    out  = run_vae(data, seed=50, train_frac=0.8, epochs=500)
    out["generated_samples"]    # (1000, n_bits) tensor of 0./1.  -> give to the MMD code
    out["test_data"]            # held-out real samples to compare against
    out["final_train_loss"]     # the loss value for the results CSV

If your framework already splits the data itself, pass the TRAINING part and train_frac=1.0.
"""

from __future__ import annotations

import csv

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from vae_model import VAE, evaluate_loss, train_vae


# ----------------------------------------------------------------------------------------
# Data helpers (same CSV format as the rest of the team: one column named "bitstring")
# ----------------------------------------------------------------------------------------
def load_bitstrings(path: str) -> torch.Tensor:
    """Read a CSV with a 'bitstring' column ("0110...") -> float tensor (N, n_bits).

    Strings are read as TEXT on purpose: reading them as numbers would silently drop
    leading zeros (that is why the team's other scripts use dtype={"bitstring": str}).
    """
    with open(path, newline="") as f:
        rows = [r["bitstring"].strip() for r in csv.DictReader(f)]
    return _parse_bitstrings(rows)


def _parse_bitstrings(rows: list[str]) -> torch.Tensor:
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(f"all bitstrings must have the same length, found lengths {sorted(lengths)}")
    return torch.tensor([[int(c) for c in r] for r in rows], dtype=torch.float32)


def to_bitstrings(samples: torch.Tensor) -> list[str]:
    """(N, n_bits) tensor of 0/1 -> list of '0110...' strings (the team's CSV format)."""
    return ["".join(str(int(b)) for b in row) for row in samples.tolist()]


def save_bitstrings(samples: torch.Tensor, path: str) -> None:
    """Write generated samples as a one-column CSV named 'bitstring'."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring"])
        for s in to_bitstrings(samples):
            w.writerow([s])


def _as_binary_tensor(data) -> torch.Tensor:
    """Accept a tensor, numpy array, list of lists, or list of '0101' strings."""
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], str):
        t = _parse_bitstrings(list(data))
    elif isinstance(data, torch.Tensor):
        t = data.detach().clone().float()
    else:
        t = torch.as_tensor(np.asarray(data), dtype=torch.float32)

    if t.ndim != 2 or t.shape[0] < 2:
        raise ValueError(f"data must be 2-D (n_samples >= 2, n_bits); got shape {tuple(t.shape)}")
    if not torch.all((t == 0) | (t == 1)):
        raise ValueError("data must contain only 0 and 1")
    return t


# ----------------------------------------------------------------------------------------
# The wrapped model
# ----------------------------------------------------------------------------------------
def run_vae(
    data,
    seed: int = 50,
    train_frac: float = 0.8,
    epochs: int = 500,
    n_samples: int = 1000,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 2.0,
    verbose: bool = False,
) -> dict:
    """Split -> train a VAE -> generate bitstrings. Everything is driven by `seed`.

    Args:
        data:       (N, n_bits) array/tensor of 0/1 (or a list of '0101' strings).
        seed:       controls the split, weight init, shuffling, and sampling. Same seed +
                    same settings -> identical outputs (on the same machine/threads).
        train_frac: fraction used for training (0 < train_frac <= 1). The split uses
                    sklearn's train_test_split(random_state=seed), the same call the
                    team's other scripts use, so seed=50 / 0.8 gives the SAME 80/20 split.
                    Use 1.0 to train on everything (no held-out set is returned).
        epochs:     number of passes over the training set.
        n_samples:  how many bitstrings to generate (default 1000, as in the plan).
        latent_dim, hidden_dim, batch_size, lr, beta: model/training settings.
                    beta weights the KL term. 1.0 is the textbook VAE; the default 2.0
                    (with latent_dim=32, epochs=500) did best on a validation split of the
                    Bars-and-Stripes TRAINING rows. Re-check it on other datasets.

    Returns a dict:
        generated_samples  float tensor (n_samples, n_bits), values 0.0/1.0
        train_loss         per-epoch average loss (negative ELBO per bitstring)
        recon_loss         per-epoch reconstruction part of the loss
        kl_loss            per-epoch KL part of the loss
        final_train_loss   train_loss[-1]
        test_loss          negative ELBO on the held-out set (None if train_frac == 1.0)
        train_data         the training split that was used
        test_data          the held-out split that was used (empty if train_frac == 1.0)
        settings           every setting used, for the results CSV
    """
    if not 0 < train_frac <= 1:
        raise ValueError("train_frac must be in (0, 1]")
    if epochs < 1 or n_samples < 1:
        raise ValueError("epochs and n_samples must be >= 1")

    x = _as_binary_tensor(data)
    n, n_bits = x.shape

    # ---- split (same call as the team's other scripts) ----
    if train_frac >= 1.0:
        train_data, test_data = x, x[:0]
    else:
        train_idx, test_idx = train_test_split(
            np.arange(n), train_size=train_frac, random_state=seed, shuffle=True
        )
        train_data, test_data = x[train_idx], x[test_idx]

    # fork_rng: we seed torch's global RNG for the weight init, then restore it afterwards,
    # so calling run_vae never disturbs the caller's own random numbers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)  # shuffling, noise, and bit sampling

        model = VAE(n_bits=n_bits, hidden_dim=hidden_dim, latent_dim=latent_dim)
        history = train_vae(model, train_data, epochs=epochs, batch_size=batch_size,
                            lr=lr, beta=beta, generator=gen, verbose=verbose)

        test_loss = evaluate_loss(model, test_data, beta, gen) if len(test_data) else None
        model.eval()
        generated = model.sample(n_samples, gen)

    return {
        "generated_samples": generated,
        "train_loss": history["total"],
        "recon_loss": history["recon"],
        "kl_loss": history["kl"],
        "final_train_loss": history["total"][-1],
        "test_loss": test_loss,
        "train_data": train_data,
        "test_data": test_data,
        "settings": {
            "model": "VAE", "seed": seed, "train_frac": train_frac, "epochs": epochs,
            "n_samples": n_samples, "latent_dim": latent_dim, "hidden_dim": hidden_dim,
            "batch_size": batch_size, "lr": lr, "beta": beta,
            "n_train": len(train_data), "n_test": len(test_data), "n_bits": n_bits,
        },
    }
