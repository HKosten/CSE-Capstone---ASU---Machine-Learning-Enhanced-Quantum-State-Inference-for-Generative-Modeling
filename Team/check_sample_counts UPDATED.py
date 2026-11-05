"""
check_sample_counts.py  --  checks that every model generates the same number of samples (task #236)

Run it from the Team folder:  python check_sample_counts.py
It takes about 3 minutes and doesn't write any files.

evaluation_framework.py compares all of a model's generated samples with all the test rows (mmd_samples in mmd.py).
How noisy an MMD score is depends on how many samples went into it, so the scores are only a fair comparison
if every model returns exactly num_samples bitstrings. This script trains each model for 1 epoch
on each dataset (same train/test split as the framework), asks it for NUM_SAMPLES bitstrings and then for 37
(to catch a model that ignores num_samples), and checks that it gets back exactly that many rows of 0/1.

When you add a model, add it to MODELS below.
If you change num_samples, seed or epsilon in Main.py, change them here too.
"""

import contextlib
import io
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import LSTM
import transformer_wrapper as tw
import vae_wrapper as vw
import qcbm_wrapper as qw

NUM_SAMPLES = 1000  # same as num_samples in Main.py
SEED = 33           # same as seed in Main.py
EPSILON = 0.3       # same as epsilon in Main.py

MODELS = {"LSTM": LSTM.LSTM, "Transformer": tw.transformer_function, "VAE": vw.vae_function, "QCBM": qw.qcbm_function}
DATASETS = {"DNA": "training_sequences_DNA.csv", "Qubit": "training_sequences_qubit.csv",
            "Sin": "sin_bitstrings.csv", "Bars+Stripes": "bars_stripes_bitstrings.csv"}

problems = []
test_rows = {}
print(f"{'Dataset':<14}{'Model':<13}{'asked for':>10}   got back")

for test_name, file in DATASETS.items():
    data = torch.tensor(pd.read_csv(file, header=None).values, dtype=torch.long)
    train_data, test_data = train_test_split(data, train_size=EPSILON, random_state=SEED)
    n_bits = data.shape[1]
    test_rows[test_name] = len(test_data)

    for model_name, model_function in MODELS.items():
        if model_name == "QCBM" and n_bits > 16:
            continue  # the QCBM only handles up to 16 bits, Main.py skips it too

        for num_samples in (NUM_SAMPLES, 37):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            settings = {"training_data": train_data, "num_epochs": 1, "seed": SEED, "epsilon": EPSILON,
                        "num_samples": num_samples, "show_output": False}

            with contextlib.redirect_stdout(io.StringIO()):  # hide the training output
                samples = model_function(**settings)

            try:
                samples = np.asarray(samples.detach().cpu() if torch.is_tensor(samples) else samples, dtype=float)
                got = f"{samples.shape[0]} rows x {samples.shape[1]} bits" if samples.ndim == 2 else f"shape {samples.shape}"
                ok = samples.shape == (num_samples, n_bits) and np.isin(samples, (0, 1)).all()
                if samples.shape == (num_samples, n_bits) and not ok:
                    got += ", but not all 0/1"
            except (ValueError, TypeError):
                got, ok = "rows of different lengths", False

            print(f"{test_name:<14}{model_name:<13}{num_samples:>10}   {got:<24}{'OK' if ok else '<-- PROBLEM'}")
            if not ok:
                problems.append(f"{model_name} on {test_name}: asked for {num_samples} x {n_bits}, got {got}")

print()
if problems:
    print(f"{len(problems)} problem(s). Every model must return exactly num_samples rows of 0/1, each as long as the dataset's rows:")
    print("\n".join("  " + p for p in problems))
    sys.exit(1)

print("All models return exactly the number of samples asked for.")
print(f"With these settings every model's MMD compares {NUM_SAMPLES} generated samples with each dataset's test rows:")
print("  " + ", ".join(f"{name} {n}" for name, n in test_rows.items()))
