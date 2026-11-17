"""QCBM fix: which ansatz / layers / starting angles / lr trains on the 12-bit qubit set? -> results/qcbm_fix/

python qcbm_fix.py --estimate
python qcbm_fix.py --settings baseline iqp
python qcbm_fix.py --settings strong2 strong3 --epochs 100 500
python qcbm_fix.py --lr-on strong2
python qcbm_fix.py --plot

same call as the framework (qcbm_function, same train/test split, 1000 samples), scored w/ three_mmds (shared mmd, default sigmas).
patience stays at the default 80, like in Main.py, so a 500 epoch run can stop early (epochs_run in the loss file).
final_train_loss = train MMD of the params that get sampled (the best step, fit() keeps those).
"""

import argparse
import contextlib
import io
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import qcbm_wrapper as qw
from qcbm import Config, QCBM, fit
from score_vs_sets import three_mmds

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"
CSV = TEAM / "training_sequences_qubit.csv"
OUT = HERE / "results" / "qcbm_fix"
COLS = ["setting", "ansatz", "layers", "init_scale", "lr", "epochs", "epsilon", "seed",
        "mmd_test", "mmd_full", "mmd_train", "train_seconds", "final_train_loss"]
# name: (ansatz, layers, init_scale, lr). init_scale 0 = qcbm.py default (strong +-pi, iqp +-0.5)
SETTINGS = {
    "baseline": ("strong", 10, 0.0, 0.05), # what Main.py runs now
    "iqp": ("iqp", 1, 0.0, 0.05), # iqp has no layers, 1 block is the whole circuit
    "strong2": ("strong", 2, 0.0, 0.05),
    "strong3": ("strong", 3, 0.0, 0.05),
    "strong10_small": ("strong", 10, 0.1, 0.05),
}
plt.rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22, "legend.fontsize": 16,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "savefig.dpi": 200, "savefig.bbox": "tight"})


def split(epsilon: float, seed: int):
    data = torch.tensor(pd.read_csv(CSV, header=None).values, dtype=torch.long)
    return train_test_split(data, train_size=epsilon, random_state=seed) # same split as the framework


def run(name: str, setting: tuple, epochs: int, epsilon: float, seed: int, n_samples: int) -> dict:
    ansatz, layers, init_scale, lr = setting
    train, _ = split(epsilon, seed)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()): # qcbm_function prints its own summary line
        gen = qw.qcbm_function(training_data=train, num_epochs=epochs, seed=seed, num_samples=n_samples,
                               ansatz=ansatz, layers=layers, lr=lr, init_scale=init_scale, verbose=False)
    secs = time.time() - t0
    hist = qw.last_history
    (OUT / "loss").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": range(1, len(hist) + 1), "loss": hist}).to_csv(OUT / "loss" / f"{name}_epochs{epochs}.csv", index=False)
    scale = init_scale or (0.5 if ansatz == "iqp" else np.pi)
    return {"setting": name, "ansatz": ansatz, "layers": layers, "init_scale": round(scale, 4), "lr": lr, "epochs": epochs,
            "epsilon": epsilon, "seed": seed, **three_mmds(gen, CSV, epsilon, seed), "train_seconds": round(secs, 1),
            "final_train_loss": min(hist), "epochs_run": len(hist)}


def save_row(row: dict):
    # 1 row per setting + epochs, a rerun replaces it
    f = OUT / "qcbm_fix.csv"
    new = pd.DataFrame([{c: row[c] for c in COLS}])
    if f.exists():
        df = pd.read_csv(f)
        df = df[~((df["setting"] == row["setting"]) & (df["epochs"] == row["epochs"]))]
        new = pd.concat([df, new], ignore_index=True)
    df = new
    df.to_csv(f, index=False)
    return df


def estimate(epsilon: float, seed: int, epochs: list):
    # 10 steps of each setting after a warm-up, times the epochs asked. sampling + scoring is ~1 s on top
    train, _ = split(epsilon, seed)
    total = 0
    for name, (ansatz, layers, init_scale, lr) in SETTINGS.items():
        cfg = Config(seed=seed, steps=12, ansatz=ansatz, layers=layers, lr=lr, init_scale=init_scale)
        with contextlib.redirect_stdout(io.StringIO()):
            m = fit(train.double(), cfg, verbose=False)
        per = m.train_seconds / len(m.history)
        total += per * sum(epochs) + 2 * len(epochs)
        print(f"{name:<16} {per:.3f} s/step -> {per * sum(epochs) + 2 * len(epochs):.0f}s for epochs {epochs}")
    print(f"total for the {len(SETTINGS)} settings: {total / 60:.1f} min (+ 1 lr 0.1 setting, at most the slowest one above)")


def plot(epochs: int):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for f in sorted((OUT / "loss").glob(f"*_epochs{epochs}.csv")):
        loss = pd.read_csv(f)
        ax.plot(loss["step"], loss["loss"], lw=3, label=f.stem.replace(f"_epochs{epochs}", ""))
    ax.set_yscale("log") # the baseline sits ~10x above the others
    ax.set(title=f"QCBM settings on Qubit (12 bits), {epochs} epochs, epsilon 0.3, seed 33",
           xlabel="training step", ylabel="training loss (train MMD, log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="setting", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / f"qcbm_fix_loss_epochs{epochs}.png")
    print(OUT / f"qcbm_fix_loss_epochs{epochs}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=[], choices=list(SETTINGS))
    ap.add_argument("--lr-on", default=None, choices=list(SETTINGS)) # same setting w/ lr 0.1
    ap.add_argument("--epochs", nargs="+", type=int, default=[100, 500])
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.estimate:
        estimate(args.epsilon, args.seed, args.epochs)

    todo = [(n, SETTINGS[n]) for n in args.settings]
    if args.lr_on:
        a, l, s, _ = SETTINGS[args.lr_on]
        todo.append((f"{args.lr_on}_lr0.1", (a, l, s, 0.1)))
    for name, setting in todo:
        for ep in args.epochs:
            row = run(name, setting, ep, args.epsilon, args.seed, args.samples)
            print(f"{name} {ep} epochs: {row['epochs_run']} run, {row['train_seconds']}s, mmd_test {row['mmd_test']:.5f}", flush=True)
            df = save_row(row)
    if todo:
        print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if args.plot:
        for ep in args.epochs:
            plot(ep)


if __name__ == "__main__":
    main()
