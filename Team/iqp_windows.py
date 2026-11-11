"""IQPopt windows: for data too wide for the QCBM (> 18 bits). new in sprint 10

python iqp_windows.py --data ../../Team/sin_bitstrings.csv --minutes 2
python iqp_windows.py --data ../../Team/training_sequences_DNA.csv --iters 200

cut the bits into overlapping windows (20 wide, 8 shared), train 1 sponsor-iqpopt IQP circuit per window, then sample window by window:
the 1st window from its own probs, each next one from its probs *given* the shared bits already drawn (exact, we have all 2^20 probs).
so neighbouring bits stay correlated, bits further apart than a window are independent.
"""

import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mmd import DEFAULT_SIGMAS

HERE = Path(__file__).resolve().parent


# ─────────────────── Layout ───────────────────
def window_starts(n_bits: int, width: int = 20, overlap: int = 8) -> list:
    if n_bits <= width:
        return [0]
    starts = list(range(0, n_bits - width + 1, width - overlap))
    if starts[-1] + width < n_bits: # last one is pushed back to the end, so it shares a bit more
        starts.append(n_bits - width)
    return starts


def idx_to_bits(idx: torch.Tensor, n: int) -> torch.Tensor:
    return (idx.unsqueeze(1) >> torch.arange(n - 1, -1, -1)) & 1 # first bit = most significant, same as qml.probs


def bits_to_idx(bits: torch.Tensor) -> torch.Tensor:
    return (bits << torch.arange(bits.shape[1] - 1, -1, -1)).sum(1)


# ─────────────────── Training ───────────────────
def load_iqpopt():
    # jax takes a few seconds to import, so only do it when a wide dataset shows up
    import jax.numpy as jnp
    import iqpopt as iqp
    import iqpopt.gen_qml as gen
    from iqpopt.utils import hardware_efficient_gates, local_gates
    return jnp, iqp, gen, hardware_efficient_gates, local_gates


def make_gates(n: int, kind: str):
    _, _, _, nn_gates, pair_gates = load_iqpopt()
    return nn_gates(n) if kind == "nn" else pair_gates(n, 2) # nn = 1 param per qubit + 1 per neighbour pair


def train_iqp(sim, x: np.ndarray, n_params: int, iters: int, seed: int, seconds: float = 0, lr: float = 0.02,
              sigmas=DEFAULT_SIGMAS, n_ops: int = 500, n_mc: int = 500, chunk: int = 50):
    """iqpopt MMD training, in chunks of `chunk` steps so the time cap can stop it. returns (params, losses).
    the adam state restarts every chunk, that is fine for this, the loss is a noisy estimate anyway"""
    jnp, iqp, gen, _, _ = load_iqpopt()
    n = x.shape[1]
    params = np.random.default_rng(seed).normal(0, 1 / np.sqrt(n), n_params) # same init as the iqpopt readme
    truth = jnp.array(x, dtype=jnp.float64)
    losses, done, t0 = [], 0, time.time()
    while done < iters:
        k = min(chunk, iters - done)
        kw = {"params": jnp.asarray(params), "iqp_circuit": sim, "ground_truth": truth,
              "sigma": list(sigmas), "n_ops": n_ops, "n_samples": n_mc}
        tr = iqp.Trainer("Adam", gen.mmd_loss_iqp, stepsize=lr)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): # trainer prints "not converged" + a progress bar every chunk
            tr.train(n_iters=k, loss_kwargs=kw, random_state=seed + done)
        params = np.asarray(tr.final_params)
        losses += [float(v) for v in tr.losses]
        done += k
        if seconds and time.time() - t0 >= seconds:
            break
    return params, losses


def fit_windows(x: np.ndarray, width: int = 20, overlap: int = 8, iters: int = 300, seed: int = 42,
                max_seconds: float = 0, gates: str = "nn", verbose: bool = False, **train_kw) -> dict:
    """one IQP circuit per window. max_seconds is for the whole thing, split evenly over the windows"""
    jnp, iqp, _, _, _ = load_iqpopt()
    n_bits = x.shape[1]
    starts = window_starts(n_bits, width, overlap)
    w = min(width, n_bits)
    glist = make_gates(w, gates)
    sim = iqp.IqpSimulator(w, glist, device="lightning.qubit") # same circuit object for every window -> jax compiles once
    per_window = max_seconds / len(starts) if max_seconds else 0

    t0, probs, curves = time.time(), [], []
    for i, s in enumerate(starts):
        params, losses = train_iqp(sim, x[:, s:s + w], len(glist), iters, seed + i, per_window, **train_kw)
        p = np.asarray(sim.probs(params), dtype=np.float64) # numpy params: jax arrays crash pennylane 0.45 + jax 0.11
        probs.append(p / p.sum())
        curves.append({"start": s, "iters": len(losses), "loss_first": losses[0], "loss_last": losses[-1]})
        if verbose:
            print(f"  window {i + 1}/{len(starts)} bits {s}-{s + w - 1}: {len(losses)} iters, "
                  f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, {time.time() - t0:.0f}s so far", flush=True)
    return {"starts": starts, "width": w, "probs": probs, "windows": curves, "n_bits": n_bits,
            "n_params_per_window": len(glist), "train_rows": len(x), "train_seconds": time.time() - t0}


# ─────────────────── Sampling ───────────────────
def sample_windows(model: dict, n_samples: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w, n_bits = model["width"], model["n_bits"]
    out = torch.zeros(n_samples, n_bits, dtype=torch.long)
    filled = 0 # columns [0, filled) are already drawn
    for s, p in zip(model["starts"], model["probs"]):
        p = torch.from_numpy(p).clamp_min(0)
        k = max(filled - s, 0) # shared bits, already drawn by the window before
        if k == 0:
            idx = torch.multinomial(p / p.sum(), n_samples, replacement=True, generator=g)
        else:
            rows = p.reshape(2 ** k, -1)[bits_to_idx(out[:, s:filled])] # (n_samples, 2^(w-k)): probs given the shared bits
            sums = rows.sum(1, keepdim=True)
            dead = sums.squeeze(1) <= 0 # shared bits this window gives 0 prob. -> uniform for the rest
            rows[dead], sums[dead] = 1.0, float(rows.shape[1])
            idx = torch.multinomial(rows / sums, 1, generator=g).squeeze(1)
        out[:, s + k:s + w] = idx_to_bits(idx, w - k)
        filled = s + w
    return out


# ─────────────────── Bitflip baseline ───────────────────
def bitflip_baseline(x: np.ndarray, n_samples: int, iters: int = 300, seed: int = 42, seconds: float = 0, gates: str = "nn"):
    """extra baseline: full width, iqpopt "bitflip" circuit = classical correlated bit flips, NOT the quantum IQP circuit.
    trains + samples at any width. returns (samples, seconds_per_iter, iters_run)"""
    _, iqp, _, _, _ = load_iqpopt()
    glist = make_gates(x.shape[1], gates)
    sim = iqp.IqpSimulator(x.shape[1], glist, bitflip=True)
    t0 = time.time()
    params, losses = train_iqp(sim, x, len(glist), iters, seed, seconds)
    per_iter = (time.time() - t0) / len(losses)
    samples = np.asarray(sim.sample(params, shots=n_samples)) % 2
    return torch.tensor(samples, dtype=torch.long), per_iter, len(losses)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--iters", type=int, default=300) # per window
    ap.add_argument("--minutes", type=float, default=0)
    args = ap.parse_args()

    from mmd import mmd_samples
    data = pd.read_csv(args.data, header=None).values
    train_x, test_x = train_test_split(data, train_size=args.epsilon, random_state=args.seed)
    model = fit_windows(train_x, iters=args.iters, seed=args.seed, max_seconds=args.minutes * 60, verbose=True)
    gen = sample_windows(model, len(test_x), args.seed)
    print(f"{len(model['starts'])} windows, {model['train_seconds']:.0f}s, "
          f"MMD vs test {mmd_samples(gen.double(), torch.tensor(test_x, dtype=torch.float64)).item():.5f}")


if __name__ == "__main__":
    main()
