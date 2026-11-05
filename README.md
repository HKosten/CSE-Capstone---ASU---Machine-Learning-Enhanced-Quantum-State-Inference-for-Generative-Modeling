## How to Run and Modify the Tests

The tests train each model on part of a dataset, have it generate new bitstrings, and score those bitstrings against the rows it never trained on, using Maximum Mean Discrepancy (MMD). Everything runs from `Team/Main.py`, which calls the shared evaluation framework in `Team/evaluation_framework.py`.

### Setup

Install Python 3.11 or newer, then install the libraries:

```
python -m pip install torch pandas numpy scikit-learn pytorch-ignite pennylane
```

The MMD comes from the `pytorch-ignite` package. Don't install the package called `ignite`: it's an unrelated library with the same import name, and the tests fail with `No module named 'ignite.metrics'` if it's installed.

### Running the Tests

Open a terminal **inside the `Team` folder** and run `Main.py` from there:

```
cd Team
python Main.py
```

Use `python3` if `python` isn't found. Running it from any other folder (for example `python Team/Main.py` from the main project folder) fails with `FileNotFoundError`, because the dataset files are looked up in the folder the program runs from. In VS Code, either open the `Team` folder itself (File > Open Folder) before pressing Run, or turn on the setting **Python › Terminal: Execute In File Dir** (`python.terminal.executeInFileDir`).

`Main.py` runs every enabled model on every dataset, one after the other. Each test prints a heading, the model's training progress, and then its score:

```
 ------------------------------------------------ Model: LSTM, Test: DNA --------------------------------------------------
...training output...
MMD: 0.000321
MMD samples: 700 generated vs 700 test rows (num_samples is 1000, but the test split only has 700 rows)
```

A full run with the default settings takes about 11 minutes on a 2-core machine, mostly the Transformer training. To check a change quickly, lower `num_epochs` (with `num_epochs` at 2, the same full run takes about 2 minutes) and comment out the tests you don't need (see below).

### What a Run Produces

Besides the terminal output, each test saves two things in the folder you ran it from (`Team`):

- **The generated bitstrings**, in a file named like `LSTM_DNA_generated_data_epsilon_0.3_epochs_100_seed_33_samples_1000` (model, dataset, then the settings). It has every bitstring the model generated, one per row, bits separated by commas, no header. It has no file extension but is plain CSV text. Running again with the same settings overwrites it.
- **The score**, as a new row in `<Dataset>_mmd_scores.csv` (for example `DNA_mmd_scores.csv`). The columns are `Model`, `MMD`, `Epsilon`, `Epochs`, `Seed`, `Samples` (how many bitstrings the model generated) and `MMD Samples` (how many of them were used for the score). Every run **adds** rows to this file instead of replacing it, so delete or rename it when you want to start a fresh table.

The results we've already uploaded are in `Team/Generated Data` and `Team/Test Scores`. New runs don't write there, so move the new files into those folders when you want to upload them.

### Changing the Test Settings

All the settings are in one dictionary near the top of `Team/Main.py`:

```python
settings = {"training_data": "training_sequences_DNA.csv", "num_epochs": 100, "seed": 33, "epsilon": 0.3, "num_samples": 1000, "show_output": False}
```

| Setting | What it controls | Default |
| --- | --- | --- |
| `training_data` | The dataset CSV, as a path relative to the `Team` folder. Each dataset section of `Main.py` sets its own. | `training_sequences_DNA.csv` |
| `num_epochs` | How long each model trains (see the note below). | `100` |
| `seed` | The random seed. The same seed gives the same train/test split and the same results. Each test is re-seeded before it starts, so turning other tests on or off doesn't change its result. | `33` |
| `epsilon` | The fraction of the dataset used for training. The rest are the test rows the MMD compares against: `0.3` of a 1000-row dataset gives 300 training rows and 700 test rows. | `0.3` |
| `num_samples` | How many bitstrings every model generates. | `1000` |
| `show_output` | `True` prints every generated bitstring to the terminal. | `False` |

The same dictionary is used for the whole file, and `Main.py` only swaps `training_data` between the dataset sections. So a change at the top applies to every model and dataset, and if you change a setting partway down the file, it stays changed for every test below that point.

`num_epochs` doesn't mean the same amount of training for every model:

- **LSTM:** one batch of 64 training rows per epoch.
- **Transformer:** one update on all the training rows per epoch.
- **VAE:** one pass over all the training rows, in batches of 64, per epoch.
- **QCBM:** one optimizer step per epoch.

### Choosing Which Models and Datasets to Test

Each test in `Main.py` is two lines: one that prints the heading, and one that runs the model.

```python
print_separators("VAE", "DNA")
ef.evaluate_model(vw.vae_function, settings, "VAE", "DNA")
```

To skip a test, put `#` at the start of both lines. To run it again, remove the `#`.

`Main.py` has a section for each of the four datasets:

| Dataset | File | Rows | Bits per row |
| --- | --- | --- | --- |
| DNA | `training_sequences_DNA.csv` | 1000 | 100 |
| Qubit | `training_sequences_qubit.csv` | 1000 | 12 |
| Sin | `sin_bitstrings.csv` | 1000 | 128 |
| Bars and Stripes | `bars_stripes_bitstrings.csv` | 1000 | 128 |

To test a different file in a section, change the `settings["training_data"]` line at the start of that section (for DNA, it's the `training_data` value in the dictionary itself). Changing one section's file doesn't change the others.

The QCBM simulates every possible bitstring (2 to the power of the number of bits), so it can only handle rows of 16 bits or fewer. Of these datasets only Qubit fits. For the others, the run prints `QCBM skipped: ...` and moves on to the next test.

### Adding a Dataset

Save the data as a CSV in the `Team` folder, in the same format as the existing files: no header row, one bitstring per row, bits separated by commas, every row the same length. Then copy one of the dataset sections in `Main.py` and change the file name and the dataset name:

```python
settings["training_data"] = "my_dataset.csv"

print_separators("LSTM", "MyDataset")
ef.evaluate_model(LSTM.LSTM, settings, "LSTM", "MyDataset")
```

The dataset name (the last argument) is used in the output file names, for example `MyDataset_mmd_scores.csv`.

### Adding a Model

The framework can test any function that takes the settings as keyword arguments and returns the generated bitstrings:

```python
def my_model_function(**settings):
    train = settings["training_data"]  # torch.long tensor of 0/1: (training rows, bits per row)
    # ...train for settings["num_epochs"] using settings["seed"]...
    # ...generate settings["num_samples"] new bitstrings...
    return samples  # (num_samples, bits per row) of 0/1, as a tensor or a list of lists
```

- `training_data` is already just the training rows (the framework has applied `epsilon`), so don't split it again.
- The function receives every setting (`training_data`, `num_epochs`, `seed`, `epsilon`, `num_samples`, `show_output`). Keep the `**settings` so it accepts the ones it doesn't use.
- It must return exactly `num_samples` rows, each as long as the dataset's rows, containing only 0s and 1s. Otherwise the test stops with an error (see below).

`vae_wrapper.py` and `qcbm_wrapper.py` are short examples of this. Then import your file at the top of `Main.py` and add a test to each dataset section:

```python
import my_model_wrapper as mw

print_separators("MyModel", "DNA")
ef.evaluate_model(mw.my_model_function, settings, "MyModel", "DNA")
```

### Understanding the Results

The MMD measures how different the generated bitstrings are from the test rows. Lower is closer, and 0 means the two sets look the same.

An MMD value depends on how many samples went into it, so the framework scores every model on the same number of samples:

- Every model must generate exactly `num_samples` bitstrings. If one returns a different number, rows of the wrong length, or values other than 0 and 1, the run stops with an error such as `LSTM returned samples with shape (300, 12), expected (1000, 12)` instead of quietly scoring that model on fewer samples.
- The MMD always compares the same number of generated bitstrings and test rows: whichever is smaller, `num_samples` or the number of test rows. With the default settings that's 700 (1000 generated, but only 700 test rows), so the score uses the first 700 generated bitstrings and all 700 test rows. The generated-data file still contains all 1000.
- The test rows are the same for every model, because the split depends only on the seed and `epsilon`.
- The terminal prints the count after each score (`MMD samples: ...`), and the scores file records it in the `MMD Samples` column.

To have the score use every generated bitstring, set `num_samples` to at most the number of test rows (700 with the default `epsilon`).

When comparing scores, only compare rows that have the same dataset, `Epsilon`, `Epochs`, `Seed` and `MMD Samples`.

### Troubleshooting

| Message | What to do |
| --- | --- |
| `FileNotFoundError: ... 'training_sequences_DNA.csv'` | Run `Main.py` from inside the `Team` folder (see Running the Tests). |
| `ModuleNotFoundError: No module named 'tkinter'` | The first line of `LSTM.py` (`from tkinter.filedialog import test`) imports something it never uses. Delete that line, or install tkinter (`sudo apt install python3-tk` on Ubuntu; on macOS with Homebrew, `brew install python-tk@3.12`, using your Python version). |
| `ModuleNotFoundError: No module named 'ignite'` or `No module named 'ignite.metrics'` | Run `python -m pip uninstall ignite`, then `python -m pip install --force-reinstall pytorch-ignite`. |
| `ModuleNotFoundError: No module named 'pennylane'` | Run `python -m pip install pennylane`. `Main.py` imports the QCBM even when its tests are commented out. |
| `QCBM skipped: 100 bits > 16-qubit simulation limit ...` | Expected. The QCBM only runs on datasets with 16 bits per row or fewer. |
| `ValueError: <Model> returned samples with shape (...), expected (...)` | That model's function didn't return exactly `num_samples` rows of the right length. Fix its wrapper (see Adding a Model). |
