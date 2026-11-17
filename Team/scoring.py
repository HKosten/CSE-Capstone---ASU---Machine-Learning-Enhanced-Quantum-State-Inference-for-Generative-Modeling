"""the 3 MMD scores per test for Team/Main.py: generated rows vs the test set, the whole dataset and the train set

copied from Khashim's Sprint 10 score_vs_sets.py (only the part Main.py uses)
"""

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mmd import mmd_samples

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
