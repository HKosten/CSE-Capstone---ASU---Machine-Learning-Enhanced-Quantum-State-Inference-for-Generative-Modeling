"""score my saved samples (results/samples) against the test set, the training set and the whole dataset -> results/score_vs_sets.json. no retraining

python score_vs_sets.py

test = held-out rows (the honest one). train = the rows the model saw, a model that memorises scores ~0 here.
whole = all 1000 rows (train + test). sqrt of the MMD, same 3 kernels as rescore.py: framework (ignite), shared default sigmas, median heuristic.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mmd import median_sigma, mmd_samples
from rescore import SETS, sqrt_mmd
from run_experiments import ignite_mmd

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"
CSV_COLS = ["model", "dataset", "epochs", "epsilon", "seed", "mmd_test", "mmd_full", "mmd_train"]


def three_mmds(gen, csv, epsilon: float, seed: int) -> dict:
    # Team/Main.py calls this after every test: shared MMD (default sigmas, not sqrt) of the generated rows vs test, whole dataset, train
    data = torch.tensor(pd.read_csv(csv, header=None).values, dtype=torch.long)
    train, test = train_test_split(data, train_size=epsilon, random_state=seed) # same split the framework made
    gen = torch.as_tensor(gen).double() # the LSTM returns a list
    return {f"mmd_{k}": mmd_samples(gen, ref.double()).item() for k, ref in (("test", test), ("full", data), ("train", train))}


def add_csv_row(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row])[CSV_COLS].to_csv(path, mode="a", index=False, header=not path.exists())


def main():
    results = {}
    for f in sorted((HERE / "results" / "runs").glob("*.json")):
        info = json.loads(f.read_text())
        data = torch.tensor(pd.read_csv(TEAM / SETS[info["dataset"]][0], header=None).values, dtype=torch.long)
        train, test = train_test_split(data, train_size=info["epsilon"], random_state=info["seed"]) # same split the run used
        gen = torch.tensor(np.load(HERE / "results" / "samples" / f"{f.stem}.npy").astype(np.int64))
        sig = median_sigma(train)
        row = {}
        for set_name, ref in (("test", test), ("train", train), ("whole", data)):
            row[set_name] = {"framework (ignite)": ignite_mmd(gen, ref), "shared default": sqrt_mmd(gen, ref, (0.5, 1.0, 2.0)),
                             f"median ({sig:.2f})": sqrt_mmd(gen, ref, (sig,))}
        results[f.stem] = row

    (HERE / "results" / "score_vs_sets.json").write_text(json.dumps(results, indent=2))
    for kern in ("framework (ignite)", "median"):
        print(f"\nkernel: {kern}   (sqrt MMD, lower = closer)")
        print(f"{'run':<34}{'test':>10}{'train':>10}{'whole':>10}")
        for name, row in results.items():
            k = [c for c in row["test"] if c.startswith(kern)][0]
            print(f"{name:<34}" + "".join(f"{row[s][k]:>10.5f}" for s in ("test", "train", "whole")))


if __name__ == "__main__":
    main()
