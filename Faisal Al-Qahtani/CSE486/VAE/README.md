# Variational Autoencoder (VAE)

**Contributor:** Faisal Al-Qahtani  
**Project:** Machine Learning-Enhanced Quantum State Inference for Generative Modeling

## Overview

This part of the project focuses on using a Variational Autoencoder (VAE) to learn patterns in binary datasets and generate new bitstrings.

Following the sponsor’s feedback, I am moving away from the Bayesian/Beta-Bernoulli approach. The goal is to develop a working generator that can be compared with the team’s other models using the same datasets and evaluation settings.

## Why VAE?

The sponsor suggested considering a VAE or Wasserstein GAN (WGAN). My comparison focuses on implementation complexity, training behavior, and support for binary outputs.

| Consideration | VAE | WGAN |
| --- | --- | --- |
| Main components | Encoder and decoder | Generator and critic |
| Training approach | Reconstruction loss and latent-space regularization | Alternating generator and critic updates |
| Binary outputs | Decoder probabilities can define Bernoulli outputs | Discrete sampling requires care during training |
| Implementation | A practical starting point for this project | Requires additional attention to adversarial training and critic constraints |

VAE is my chosen direction because it provides a straightforward way to model binary data and fits the time available for implementation. This is an implementation decision, not a claim that it will outperform the other models.

## Model Design

1. **Encoder:** Takes a binary input and produces the parameters (mean and log-variance) of a latent Gaussian.
2. **Latent sampling:** Samples a latent vector with the reparameterization trick during training.
3. **Decoder:** Converts the latent vector into one probability (logit) per output bit.
4. **Binary generation:** Draws `z ~ N(0, I)`, decodes it, and samples each bit from its predicted Bernoulli probability.

Training minimizes the negative ELBO: binary cross-entropy reconstruction loss + `beta` × KL divergence.

Settings used (all are arguments of `run_vae`): one hidden layer of 256 units in the encoder and decoder, 32 latent dimensions, `beta = 2.0`, Adam (lr 1e-3), batch size 64, 500 epochs. These were picked from a small sweep that used a validation split carved only from the *training* rows, so the held-out set was never used for tuning. They were tuned on Bars-and-Stripes only.

## Dataset

The initial dataset is the Bars-and-Stripes dataset prepared for the project:

- Grid size: 8 × 16.
- Bitstring length: 128.
- Number of unique samples: 1,000 (762 vertical-bar type, 236 horizontal-stripe type, 2 all-0/all-1).
- Dataset-generation seed: 50.
- Patterns: Horizontal stripes or vertical bars.

The split is 80/20 using `train_test_split(random_state=seed)`, the same call the other scripts use, so seed 50 gives the **same** 800/200 split as the Bayesian and transformer scripts.

## Files

| File | Purpose |
| --- | --- |
| `vae_model.py` | The `VAE` class, the loss, the training loop, and sampling |
| `vae_interface.py` | `run_vae(...)`: the full wrapped model (split, train, generate, losses, settings), plus CSV helpers. Kept separate from the model so it is easy to change |
| `vae_wrapper.py` | `vae_function(trainingData)`: the team-format wrapper, with the same shape as the transformer’s `transformer_function(trainingData)` |
| `vae_wrapper_test.py` | Quick check of the wrapper: runs it on a CSV, checks the output format, prints the generated bitstrings |
| `run_vae_experiment.py` | Trains, generates, computes MMD and baselines, checks reproducibility, appends a row to `vae_results.csv` |
| `Sprint8_VAE_Bars_Stripes.ipynb` | Google Colab notebook that runs all of the above and shows the plots and results |
| `vae_results.csv` | Created when you run the experiment (one row per run) |

## How to Run

Requirements: `torch`, `numpy`, `scikit-learn` (tested on Google Colab with PyTorch 2.11, and in two other environments with PyTorch 2.14). `pytorch-ignite` is optional (only used to report the team’s ignite MMD).

```bash
cd "Faisal Al-Qahtani/CSE486/VAE"
python run_vae_experiment.py                      # seed 50, Bars-and-Stripes (roughly 1 minute on one CPU core, including the reproducibility check)
python run_vae_experiment.py --seeds 50 51 52     # several splits
python run_vae_experiment.py --data other.csv --epochs 300 --save-samples generated.csv
python vae_wrapper_test.py path/to/bitstrings.csv  # checks the team-format wrapper and prints generated bitstrings
```

By default the experiment reads `../Bayesian Model/bars_stripes_128.csv`. Any CSV with a `bitstring` column works.

**On Google Colab:** open `Sprint8_VAE_Bars_Stripes.ipynb`, choose `Runtime → Run all`, and upload `bars_stripes_128.csv` when asked. The notebook writes the `.py` files itself, so nothing else has to be uploaded.

## Interface for the Evaluation Framework

The model is wrapped in a function so the shared framework can call it like the other models. `vae_function` follows the format of the transformer wrapper (`transformer_function(trainingData)`): one input, and it returns the bitstrings the model generates.

```python
from vae_wrapper import vae_function

output = vae_function("bitstrings.csv")      # CSV with a "bitstring" column
for r in output:
    print("".join(map(str, r.tolist())))     # one generated bitstring per line
```

- **Input:** `trainingData` is a path to a CSV with a `bitstring` column (or an array, tensor, or list of `"0101..."` strings). Optional keyword arguments: `seed` (default 50), `epochs` (500), `n_samples`, and the model settings (`latent_dim`, `hidden_dim`, `batch_size`, `lr`, `beta`).
- **Output:** a `torch.long` tensor of shape `(n_samples, n_bits)` with values 0/1, the same dtype and layout as the transformer wrapper. By default there is one generated bitstring per input row, so the output can be compared with the input in ignite’s MMD, which needs two sets of the same size.
- It trains on **every** row it is given, so the framework should pass the training split. The same seed gives identical output.
- Checked with the ignite MMD call used in the team’s testing framework, and on both the Bars-and-Stripes data and the transformer’s `bitstrings.csv`.

For more detail (loss curves, held-out data, the settings used), call `run_vae` in `vae_interface.py`:

```python
from vae_interface import load_bitstrings, run_vae

data = load_bitstrings("bars_stripes_128.csv")             # (N, n_bits) tensor of 0/1
out = run_vae(data, seed=50, train_frac=0.8, epochs=500)   # all other settings have defaults

out["generated_samples"]   # (1000, n_bits) tensor of 0./1.  -> pass to the MMD code
out["test_data"]           # held-out real samples to compare against
out["final_train_loss"]    # loss for the results CSV (also out["test_loss"], out["train_loss"] per epoch)
out["settings"]            # every setting used
```

- `run_vae` does the train/test split itself (`train_frac`, using the same `train_test_split(random_state=seed)` call as the other scripts) and does not disturb the caller’s global random state.
- The exact format the framework expects is still the team’s decision. If it changes, only `vae_wrapper.py` and `vae_interface.py` need to change.

## Evaluation

Generated samples are compared with the held-out data using MMD. The same comparison is run for a baseline of 1,000 random bitstrings. The target is a lower MMD than the random baseline.

Two MMD versions are reported because the choice of kernel bandwidth matters a lot at 128 bits:

- **MMD² (gaussian):** unbiased, bandwidth taken from the data (median heuristic, 3 scales). It can be slightly negative, which just means about 0.
- **MMD (ignite, `var=1.0`):** the setting the other scripts use. Sample sets are trimmed to equal size, as ignite requires. Note that ignite returns the *square root* of MMD², so its numbers are not directly comparable with MMD² values. It also clamps a negative estimate to exactly 0, so `0.00000` (seed 54 in the Colab run) means the VAE samples and the held-out data were indistinguishable at that kernel width.

Bars-and-Stripes also has an exact validity check: a sample is valid if every row is constant (stripes) or every column is constant (bars). The script reports the share of valid samples and the average number of bits (out of 128) away from the nearest valid pattern.

## Results

Seeds 50–54 (five different 80/20 splits), 500 epochs, run on Google Colab (CPU, PyTorch 2.11) with `Sprint8_VAE_Bars_Stripes.ipynb`. The “floor” is real training rows vs. held-out rows (what a perfect generator would score).

| Seed | MMD² VAE | MMD² random | Bits wrong, VAE / random | Valid patterns |
| --- | --- | --- | --- | --- |
| 50 | 0.00031 | 0.00423 | 13.5 / 46.1 | 0.6% |
| 51 | 0.00143 | 0.00549 | 14.0 / 45.9 | 0.0% |
| 52 | 0.00028 | 0.00445 | 15.0 / 46.1 | 0.5% |
| 53 | −0.00018 | 0.00423 | 15.0 / 46.2 | 0.6% |
| 54 | 0.00127 | 0.00466 | 14.3 / 46.2 | 0.5% |
| **Mean** | **0.00062** | **0.00461** | **14.4 / 46.1** | **0.4%** |

- The VAE beat the random baseline on MMD² in 5 of 5 runs. The mean floor is −0.00019 (about 0).
- Final training loss was 35.6–36.8 and held-out loss 37.6–39.3 (negative ELBO per bitstring), so there is no large overfitting gap.
- Reproducibility: running seed 50 twice gave identical samples and identical loss curves (same machine and settings). Different PyTorch versions give slightly different numbers: in another environment with PyTorch 2.14 the same code gave a mean VAE MMD² of 0.00067 instead of 0.00062 (the random baseline and the floor were identical), with the same conclusions.

## Limitations

- **Samples are noisy.** They have the right overall structure (about 14 of 128 bits wrong on average, versus about 46 for random bits), but fewer than 1% are exactly valid patterns. In an earlier diagnostic run with different settings (`beta=1`, 16 latent dims, 200 epochs), reconstructions of held-out data were mostly valid while samples drawn from `N(0, I)` mostly were not, which points to generation from the prior rather than to learning, but this was not studied further.
- **How good the VAE looks depends on the MMD kernel, so the team should agree on one setting.** On the seed-50 split, the VAE's MMD² is about 73% of the random baseline's under ignite `var=1.0` (compared after squaring ignite's output) and about 7% with bandwidths scaled to the data. Under the default sigmas (0.5, 1, 2) in Khashim's `mmd.py` it was about 23% in an earlier run with PyTorch 2.14. All three rank the VAE better than random, but the narrow kernels see little of the structure at 128 bits, because typical pairs of real strings are about 8 apart (Euclidean distance). This is one seed on one dataset, and the exact values shift a little between PyTorch versions.
- `beta = 2.0` is not the textbook VAE (`beta = 1.0`). It did best on the validation split, but defaults were tuned on Bars-and-Stripes only; re-check on other datasets.
- Reproducibility holds under the same execution settings (CPU, same PyTorch version).

## Current Status

VAE is implemented, tested on Bars-and-Stripes with five seeds (run on Google Colab with `Sprint8_VAE_Bars_Stripes.ipynb`), and wrapped in `vae_function(trainingData)` in the team’s wrapper format. Next: confirm with the team that the framework works with it, then run it through the shared framework on the other datasets (DNA, sine-wave bitstrings).
