"""poster plots from the Team/Main.py sweep: loss graph per model + MMD bar charts -> results/plots/*.png

python plots.py
python plots.py --dir results

reads <dir>/mmd_three_sets.csv (1 row per test) and <dir>/loss/*.csv (1 file per test: step, loss). Main.py writes both and calls this at the end.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # no window, only png files
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
POSTER = HERE / "results"
SCORES = "mmd_three_sets.csv"
BARS = {"mmd_test": ("test set", "#1f77b4"), "mmd_full": ("whole dataset", "#7f7f7f"), "mmd_train": ("train set", "#ff7f0e")}
plt.rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22, "legend.fontsize": 18,
                     "xtick.labelsize": 20, "ytick.labelsize": 18, "savefig.dpi": 200, "savefig.bbox": "tight"})


# ─────────────────── Loss ───────────────────
def run_stem(model: str, dataset: str, epochs: int, epsilon: float, seed: int) -> str:
    return f"{model}_{dataset}_epochs{epochs}_eps{epsilon}_seed{seed}"


def parse_loss(text: str) -> list:
    # every model prints "Epoch N, Loss: x" while it trains, so the printed lines are the loss history
    return [(int(s), float(v)) for s, v in re.findall(r"Epoch (\d+), Loss: ([-+.\deE]+)", text)]


def save_loss(out_dir, stem: str, points: list):
    (Path(out_dir) / "loss").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(points, columns=["step", "loss"]).to_csv(Path(out_dir) / "loss" / f"{stem}.csv", index=False)


def plot_loss(out_dir=POSTER) -> list:
    """1 png per model, 1 panel per dataset, 1 line per epsilon. uses the longest run (most epochs) and the first seed"""
    out_dir, made = Path(out_dir), []
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted((out_dir / "loss").glob("*.csv")):
        m = re.fullmatch(r"(.+?)_(.+)_epochs(\d+)_eps([\d.]+)_seed(\d+)", f.stem)
        if m and len(pd.read_csv(f)):
            runs.append({"model": m[1], "dataset": m[2], "epochs": int(m[3]), "epsilon": float(m[4]), "seed": int(m[5]), "file": f})
    runs = pd.DataFrame(runs)
    for model, r in (runs.groupby("model") if len(runs) else []):
        datasets = list(dict.fromkeys(r["dataset"]))
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6.5), squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            d = r[r["dataset"] == ds]
            d = d[(d["epochs"] == d["epochs"].max()) & (d["seed"] == d["seed"].min())]
            for _, run in d.sort_values("epsilon").iterrows():
                loss = pd.read_csv(run["file"])
                ax.plot(loss["step"], loss["loss"], lw=3, marker="o" if len(loss) < 30 else None, ms=8, label=f"epsilon {run['epsilon']}")
            ax.set(title=f"{model} on {ds}", xlabel="training step", ylabel="training loss")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.savefig(out_dir / "plots" / f"loss_{model}.png")
        plt.close(fig)
        made.append(out_dir / "plots" / f"loss_{model}.png")
    return made


# ─────────────────── Bars ───────────────────
def plot_bars(out_dir=POSTER) -> list:
    """1 png per dataset + epochs + epsilon: 1 group per model, 3 bars each (test, whole, train). mean over the seeds"""
    out_dir, made = Path(out_dir), []
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out_dir / SCORES)
    for (ds, epochs, eps), d in df.groupby(["dataset", "epochs", "epsilon"]):
        mean = d.groupby("model", sort=False)[list(BARS)].mean()
        fig, ax = plt.subplots(figsize=(3.2 * len(mean) + 3, 7))
        for i, (col, (label, color)) in enumerate(BARS.items()):
            bars = ax.bar([x + (i - 1) * 0.27 for x in range(len(mean))], mean[col], width=0.27, label=label, color=color)
            ax.bar_label(bars, fmt="%.2g", fontsize=13, padding=3)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(mean)), mean.index)
        seeds = sorted(d["seed"].unique())
        ax.set(xlabel="model", ylabel="MMD vs real data (lower = closer)",
               title=f"{ds}: {epochs} epochs, epsilon {eps}, seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))}")
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="generated samples vs", loc="upper left", bbox_to_anchor=(1.01, 1)) # outside, so it never covers a bar
        name = out_dir / "plots" / f"mmd_bars_{ds}_epochs{epochs}_eps{eps}.png"
        fig.savefig(name)
        plt.close(fig)
        made.append(name)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(POSTER))
    args = ap.parse_args()
    for f in plot_loss(args.dir) + plot_bars(args.dir):
        print(f)


if __name__ == "__main__":
    main()
