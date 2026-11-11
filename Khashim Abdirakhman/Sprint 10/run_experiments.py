"""Sprint 10 runs: 6 QCBM runs on the 12-bit qubit set + 1 IQP-windows run per wide set -> results/runs/*.json, results/samples/*.npy

python run_experiments.py --epochs 3000 --only qubit
python run_experiments.py --only wide
python run_experiments.py --only wide --datasets sin --iters 100 --cap 120
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import iqp_windows as iw
from mmd import mmd_samples
from qcbm_wrapper import qcbm_function

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"
WIDE = {"dna": "training_sequences_DNA.csv", "sin": "sin_bitstrings.csv", "bas": "bars_stripes_bitstrings.csv"}


def ignite_mmd(gen: torch.Tensor, real: torch.Tensor) -> float:
    # same call as Team/evaluation_framework.py, so this is the number the team sees
    from ignite.metrics import MaximumMeanDiscrepancy
    count = min(len(gen), len(real))
    m = MaximumMeanDiscrepancy()
    m.update((gen[:count].float(), real[:count].float()))
    return m.compute()


def score(gen: torch.Tensor, test: torch.Tensor) -> dict:
    return {"mmd_framework_ignite": ignite_mmd(gen, test),
            "mmd2_shared_default_sigmas": mmd_samples(gen.double(), test.double()).item()}


def save(name: str, gen: torch.Tensor, info: dict):
    (HERE / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "samples").mkdir(parents=True, exist_ok=True)
    np.save(HERE / "results" / "samples" / f"{name}.npy", gen.numpy().astype(np.uint8))
    (HERE / "results" / "runs" / f"{name}.json").write_text(json.dumps(info, indent=2))


def split(csv: str, epsilon: float, seed: int):
    data = torch.tensor(pd.read_csv(TEAM / csv, header=None).values, dtype=torch.long)
    return train_test_split(data, train_size=epsilon, random_state=seed) # same split as the framework


def run_qubit(args):
    for eps in (0.1, 0.5, 0.9):
        for seed in (33, 34):
            name = f"qubit_qcbm_eps{eps}_seed{seed}"
            train_x, test_x = split("training_sequences_qubit.csv", eps, seed)
            t0 = time.time()
            gen = qcbm_function(training_data=train_x, num_epochs=args.epochs, seed=seed, num_samples=len(test_x),
                                max_seconds=args.cap, patience=args.epochs)
            secs = time.time() - t0
            info = {"dataset": "qubit", "model": "QCBM", "epsilon": eps, "seed": seed, "n_train": len(train_x),
                    "n_test": len(test_x), "epochs_asked": args.epochs, "cap_seconds": args.cap,
                    "hit_cap": secs >= args.cap, "seconds": round(secs, 1), **score(gen, test_x)}
            save(name, gen, info)
            print(f"{name}: {info['mmd_framework_ignite']:.4f} framework MMD, {secs:.0f}s{' (hit cap)' if info['hit_cap'] else ''}", flush=True)


def run_wide(args):
    for key in args.datasets:
        csv = WIDE[key]
        train_x, test_x = split(csv, 0.5, 33)
        t0 = time.time()
        gen = qcbm_function(training_data=train_x, num_epochs=args.iters, seed=33, num_samples=len(test_x), max_seconds=args.cap)
        secs = time.time() - t0
        n_win = len(iw.window_starts(train_x.shape[1]))
        info = {"dataset": key, "model": "IQP windows (iqpopt)", "epsilon": 0.5, "seed": 33, "n_train": len(train_x),
                "n_test": len(test_x), "n_bits": train_x.shape[1], "windows": n_win, "width": 20, "overlap": 8,
                "iters_per_window": args.iters, "cap_seconds": args.cap, "seconds": round(secs, 1),
                "hit_cap": secs >= args.cap, **score(gen, test_x)}
        save(f"{key}_iqpwindows_eps0.5_seed33", gen, info)
        print(f"{key} windows: {info['mmd_framework_ignite']:.5f} framework MMD, {secs:.0f}s", flush=True)

        if args.no_bitflip:
            continue
        t0 = time.time()
        bf, per_iter, n_it = iw.bitflip_baseline(train_x.numpy(), len(test_x), iters=args.iters, seed=33, seconds=args.bitflip_cap)
        info = {"dataset": key, "model": "iqpopt bitflip (classical, extra baseline)", "epsilon": 0.5, "seed": 33,
                "n_train": len(train_x), "n_test": len(test_x), "n_bits": train_x.shape[1], "iters": n_it,
                "seconds_per_iter": round(per_iter, 3), "seconds": round(time.time() - t0, 1), **score(bf, test_x)}
        save(f"{key}_bitflip_eps0.5_seed33", bf, info)
        print(f"{key} bitflip: {info['mmd_framework_ignite']:.5f} framework MMD, {n_it} iters at {per_iter:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["qubit", "wide", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=2000) # qubit: plateau point from epoch_test.py
    ap.add_argument("--iters", type=int, default=500) # wide: iqpopt steps per window
    ap.add_argument("--cap", type=float, default=600) # seconds per run
    ap.add_argument("--bitflip-cap", type=float, default=120)
    ap.add_argument("--no-bitflip", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(WIDE), choices=list(WIDE))
    args = ap.parse_args()
    if args.only in ("qubit", "all"):
        run_qubit(args)
    if args.only in ("wide", "all"):
        run_wide(args)


if __name__ == "__main__":
    main()
