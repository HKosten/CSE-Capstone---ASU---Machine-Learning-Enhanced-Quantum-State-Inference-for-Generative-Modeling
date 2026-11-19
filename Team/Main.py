import evaluation_framework as ef
import LSTM
import transformer_wrapper as tw
import vae_wrapper as vw
import qcbm_wrapper as qw
import numpy as np
import contextlib
import io
import sys
from pathlib import Path

# the 3 MMD scores per test (scoring.py) and the loss graphs + bar charts (plots.py)
import scoring
import plots
import iqpopt_wrapper as iqw

RESULTS = Path(__file__).resolve().parent / "results" # mmd_three_sets.csv, loss/*.csv, plots/*.png

count = 1
def print_separators(model_name, settings):
    global count
    print(f"\n --------------------------------- Test #{count} | Model: {model_name} | dataset: {settings['training_data']} | epochs count: {settings['num_epochs']} | seed: {settings['seed']} | epsilon: {settings['epsilon']:.1f} | sample count: {settings['num_samples']} | show_output: {settings['show_output']} | ----------------------------------- \n")
    count += 1

def run_qcbm(settings, test_name):
    """
    the QCBM simulates 2^n numbers, so one circuit only handles up to 18 bits (the 12-bit qubit set).
    wider data runs as overlapping 20-bit iqpopt windows (iqp_windows.py), see Sprint 10 HOWTO for the packages it needs.
    a ValueError, or jax/iqpopt not installed, is printed and the other models keep going
    """
    try:
        ef.evaluate_model(qw.qcbm_function, settings, "QCBM", test_name)
    except (ValueError, ImportError) as e:
        print(f"QCBM skipped: {e}")

class Tee:
    # prints like normal and keeps a copy, the loss graphs are read from the "Epoch N, Loss: x" lines
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()

def run_scored(model, settings, model_name, test_name):
    """
    1 test through the framework like before, then the same generated rows are scored against the test set, the whole dataset and the train set.
    adds 1 row to mmd_three_sets.csv and saves the loss history of the run (both in RESULTS)
    """
    kept, log = {}, io.StringIO()
    def keep(**model_settings):
        kept["gen"] = model(**model_settings)
        return kept["gen"]

    print_separators(model_name, settings)
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            ef.evaluate_model(keep, settings, model_name, test_name)
    except (ValueError, ImportError) as e: # same as run_qcbm
        if model_name not in ("QCBM", "IQPopt"):
            raise
        print(f"{model_name} skipped: {e}")
        return

    row = {"model": model_name, "dataset": test_name, "epochs": settings["num_epochs"], "epsilon": settings["epsilon"], "seed": settings["seed"],
           **scoring.three_mmds(kept["gen"], settings["training_data"], settings["epsilon"], settings["seed"])}
    scoring.add_csv_row(RESULTS / plots.SCORES, row)
    print(f"MMD vs test: {row['mmd_test']:.6f} | whole dataset: {row['mmd_full']:.6f} | train: {row['mmd_train']:.6f}")

    # the QCBM and IQPopt keep the loss of every step, the other models only print it every 50 epochs
    full = {"QCBM": qw, "IQPopt": iqw}.get(model_name)
    loss = list(enumerate(full.last_history, 1)) if full and full.last_history else plots.parse_loss(log.getvalue())
    plots.save_loss(RESULTS, plots.run_stem(model_name, test_name, settings["num_epochs"], settings["epsilon"], settings["seed"]), loss)

def qcbm_small(**model_settings):
    # above 18 bits qcbm_function runs the same iqpopt windows as the IQPopt model, so in the sweep the QCBM only does the data its own circuit fits
    n_bits = model_settings["training_data"].shape[1]
    if n_bits > 18:
        raise ValueError(f"{n_bits} bits > 18-qubit simulation limit, see the IQPopt rows for this dataset")
    return qw.qcbm_function(**model_settings)


# EPSILON TEST: varies epsilon from 0.1 to 0.9, incrementing by 0.1 each time
def epsilon_test(model, settings, model_name, test_name):
    
    test_settings = dict(settings) # copy, so the tests after this still get the original settings

    for i in np.arange(0.1, 1.0, 0.1):
        test_settings["epsilon"] = round(float(i), 1) # np.arange gives 0.30000000000000004
        print_separators(model_name, test_settings)
        ef.evaluate_model(model, test_settings, model_name, test_name)


# EPOCHS TEST: varies num_epochs from 0 to 10000, incrementing every 100 for now to test for flattening
def epochs_test(model, settings, model_name, test_name):
    
    test_settings = dict(settings) # copy, so the tests after this still get the original settings
    
    for i in range(0, 10000, 100):
        test_settings["num_epochs"] = i
        print_separators(model_name, test_settings)
        ef.evaluate_model(model, test_settings, model_name, test_name)


settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 1000, "show_output": False}

# VARYING EPSILON:
# All other original settings remain the same except epsilon
# Every possible model passed in and tested. 
# Currently on the DNA dataset, change the filename in the settings dictionary and the curr_dataset_name to change
models = {"LSTM": LSTM.LSTM, "Transformer":tw.transformer_function, "VAE":vw.vae_function}
curr_dataset_name = "DNA"
"""
for model_name, model in models.items():
    epsilon_test(model, settings, model_name, curr_dataset_name)
"""

# SWEEP (sponsor, Nov 5): several settings in 1 run. every model on every dataset for every epochs x epsilon x seed below.
# every test adds 1 row to the same csv (RESULTS/mmd_three_sets.csv), the loss graphs and bar charts are made at the end
EPOCHS = [50, 100, 1000]
EPSILONS = [0.3, 0.7]
SEEDS = [33]
DATASETS = {"DNA": "training_sequences_DNA.csv", "Qubit": "training_sequences_qubit.csv", "Sin": "sin_bitstrings.csv",
            "Bars+Stripes": "bars_stripes_bitstrings.csv"}
sweep_models = {**models, "QCBM": qcbm_small, "IQPopt": iqw.iqpopt_function}

for test_name, csv in DATASETS.items():
    for seed in SEEDS:
        for num_epochs in EPOCHS:
            for epsilon in EPSILONS:
                test_settings = {**settings, "training_data": csv, "num_epochs": num_epochs, "epsilon": epsilon, "seed": seed}
                for model_name, model in sweep_models.items():
                    run_scored(model, test_settings, model_name, test_name)

for f in plots.plot_loss(RESULTS) + plots.plot_bars(RESULTS):
    print(f"saved {f}")


# the older tests are switched off so 1 run of this file is the sweep above. take the quotes away to run them again
"""
# VARYING NUM EPOCHS:
# All other original settings remain the same except num_epochs
for model_name, model in models.items():
    epochs_test(model, settings, model_name, curr_dataset_name)

# Potential new way to run tests?
for model_name, model in models.items():
    print_separators(model_name, settings)
    ef.evaluate_model(model, settings, model_name, curr_dataset_name)
run_qcbm(settings, curr_dataset_name)

# ORIGINAL TESTS
print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "DNA")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "DNA")

print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "DNA")
print_separators("QCBM", settings)
run_qcbm(settings, "DNA")

settings["training_data"] = "training_sequences_qubit.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Qubit")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Qubit")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Qubit")
print_separators("QCBM", settings)
run_qcbm(settings, "Qubit")

settings["training_data"] = "sin_bitstrings.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Sin")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Sin")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Sin")
print_separators("QCBM", settings)
run_qcbm(settings, "Sin")

settings["training_data"] = "bars_stripes_bitstrings.csv"

print_separators("LSTM", settings)
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "Bars+Stripes")
print_separators("Transformer", settings)
ef.evaluate_model(tw.transformer_function, settings, "Transformer", "Bars+Stripes")
print_separators("VAE", settings)
ef.evaluate_model(vw.vae_function, settings, "VAE", "Bars+Stripes")
print_separators("QCBM", settings)
run_qcbm(settings, "Bars+Stripes")
"""
