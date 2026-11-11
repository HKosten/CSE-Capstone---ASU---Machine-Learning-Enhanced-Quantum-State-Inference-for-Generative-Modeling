"""Rescore saved samples with different kernel widths + validity / correlation checks -> results/rescore.json. no retraining

python rescore.py
python rescore.py --datasets dna sin

every sample set (team models from Team/Generated Data, my runs from results/samples, 2 references) is scored against the
same 500 test rows (epsilon 0.5, seed 33 split). Those rows are in nobody's training set: the team models trained on 30%,
and with this seed that 30% sits inside the other 50%.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mmd import iqpopt_sigmas, median_sigma, mmd_samples
from run_experiments import ignite_mmd

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"
SETS = {  # key: (csv, name in Team/Generated Data)
    "qubit": ("training_sequences_qubit.csv", "Qubit"), "dna": ("training_sequences_DNA.csv", "DNA"),
    "sin": ("sin_bitstrings.csv", "Sin"), "bas": ("bars_stripes_bitstrings.csv", "Bars+Stripes")}


# ─────────────────── Scores ───────────────────
def kernels(train: torch.Tensor) -> dict:
    # name -> sigmas for mmd_samples. the framework (ignite) one is separate, it has its own formula
    return {"shared default (0.5, 1, 2)": (0.5, 1.0, 2.0),
            f"median heuristic ({median_sigma(train):.2f})": (median_sigma(train),),
            "iqpopt sigma_heuristic x3 (" + ", ".join(f"{s:.1f}" for s in iqpopt_sigmas(train, 3)) + ")": tuple(iqpopt_sigmas(train, 3))}


def sqrt_mmd(gen, test, sigmas) -> float:
    return max(mmd_samples(gen.double(), test.double(), sigmas).item(), 0.0) ** 0.5 # sqrt so it reads like the framework number


# ─────────────────── Checks ───────────────────
def valid_grid(rows: np.ndarray, h: int = 8, w: int = 16) -> np.ndarray:
    g = rows.reshape(len(rows), h, w) # bars and stripes: every row constant, or every column constant
    return (g == g[:, :, :1]).all(axis=(1, 2)) | (g == g[:, :1, :]).all(axis=(1, 2))


def adjacent_corr(rows: np.ndarray) -> float:
    x = rows.astype(float)
    c = np.nan_to_num(np.corrcoef(x.T)) # constant columns give nan -> 0
    return float(np.mean([abs(c[i, i + 1]) for i in range(x.shape[1] - 1)]))


def checks(key: str, gen: np.ndarray, real_all: np.ndarray, real_train: np.ndarray) -> dict:
    out = {"bit_mean_error": float(np.abs(gen.mean(0) - real_train.mean(0)).mean()),
           "adjacent_corr": adjacent_corr(gen), "adjacent_corr_real": adjacent_corr(real_train)}
    d = lambda a: float((a[:, None, :] != a[None, :, :]).sum(-1).mean() * len(a) / (len(a) - 1))
    out["mean_hamming"], out["mean_hamming_real"] = d(gen[:300]), d(real_train[:300])
    if key == "bas":
        out["valid_rows"] = float(valid_grid(gen).mean())
    elif key in ("sin", "qubit"): # sin = 128 shifted square waves, qubit = 92 distinct rows: share of rows seen in the real data
        seen = {r.tobytes() for r in real_all.astype(np.uint8)}
        out["valid_rows"] = float(np.mean([r.tobytes() in seen for r in gen.astype(np.uint8)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(SETS), choices=list(SETS))
    args = ap.parse_args()

    results = {}
    for key in args.datasets:
        csv, team_name = SETS[key]
        data = pd.read_csv(TEAM / csv, header=None).values
        train, test = train_test_split(torch.tensor(data, dtype=torch.long), train_size=0.5, random_state=33)
        n = len(test)
        sample_sets = {}
        for model in ("LSTM", "Transformer", "VAE", "QCBM"):
            f = TEAM / "Generated Data" / f"{model}_{team_name}_generated_data_epsilon_0.3_epochs_100_seed_33_samples_1000"
            if f.exists():
                sample_sets[f"{model} (team, 30% train)"] = torch.tensor(pd.read_csv(f, header=None).values[:n])
        for f in sorted((HERE / "results" / "samples").glob(f"{key}_*eps0.5_seed33.npy")) + \
                sorted((HERE / "results" / "samples").glob(f"{key}_qcbm_eps0.5_seed33.npy")):
            sample_sets[f"{f.stem.split('_')[1]} (sprint 10, 50% train)"] = torch.tensor(np.load(f).astype(np.int64))[:n]
        g = torch.Generator().manual_seed(0)
        sample_sets["REF coin flips"] = torch.randint(0, 2, (n, train.shape[1]), generator=g)
        sample_sets["REF copy of training rows"] = train[:n]

        ks = kernels(train)
        rows = {}
        for name, s in sample_sets.items():
            r = {"framework (ignite, var=1)": ignite_mmd(s, test)}
            r.update({k: sqrt_mmd(s, test, sig) for k, sig in ks.items()})
            r["checks"] = checks(key, s.numpy(), data, train.numpy())
            rows[name] = r
        results[key] = rows

        cols = ["framework (ignite, var=1)"] + list(ks)
        print(f"\n=== {key}: {train.shape[1]} bits, test {n} rows (sqrt MMD, lower = closer)")
        print(f"{'':<36}" + "".join(f"{c[:26]:>28}" for c in cols))
        for name, r in rows.items():
            print(f"{name:<36}" + "".join(f"{r[c]:>28.5f}" for c in cols))
        print(f"{'':<36}{'bit-mean err':>14}{'adj corr':>10}{'(real)':>9}{'hamming':>10}{'(real)':>9}{'valid':>8}")
        for name, r in rows.items():
            c = r["checks"]
            print(f"{name:<36}{c['bit_mean_error']:>14.3f}{c['adjacent_corr']:>10.3f}{c['adjacent_corr_real']:>9.3f}"
                  f"{c['mean_hamming']:>10.1f}{c['mean_hamming_real']:>9.1f}{c.get('valid_rows', float('nan')):>8.1%}")
    (HERE / "results" / "rescore.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
