"""
run_vae_experiment.py  --  train the VAE and evaluate it, following the plan in README.md.

For each seed it:
  1. splits the data, trains the VAE, generates 1,000 bitstrings           (vae_interface.run_vae)
  2. computes MMD between the generated samples and the HELD-OUT real samples
  3. computes the same MMD for a baseline of 1,000 random bitstrings        (target: VAE < random)
  4. computes MMD for real training rows vs held-out                        (what "perfect" looks like)
  5. (Bars-and-Stripes only) checks how many generated samples are valid patterns, and how
     many bits they are away from the nearest valid pattern
  6. checks reproducibility: the same seed run twice must give identical output
  7. appends one row to vae_results.csv

Run it from this folder:
    python run_vae_experiment.py                       # seed 50, Bars-and-Stripes data
    python run_vae_experiment.py --seeds 50 51 52      # several splits
    python run_vae_experiment.py --data other.csv --epochs 300 --save-samples out.csv

About the MMD numbers (important when comparing with teammates)
---------------------------------------------------------------
MMD depends on the kernel's bandwidth. On 128-bit strings, two random strings differ in ~64
bits, so a narrow kernel (like ignite's default var=1.0) treats almost every pair as
"completely different" and cannot tell a good generator from random noise. So this script
reports two versions:
  * "MMD2 (gaussian)":  unbiased MMD^2, bandwidth set from the data (median heuristic),
                        averaged over 3 scales. Can be slightly negative (that means ~0).
  * "MMD (ignite)":     ignite's MaximumMeanDiscrepancy with the team's default var=1.0,
                        so numbers are comparable with Jacqui's / the Bayesian scripts.
                        (ignite needs equal-size sets, so samples are trimmed to match.)
When the team agrees on ONE shared MMD for the evaluation framework, swap it in here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from vae_interface import load_bitstrings, run_vae, save_bitstrings

# The Bars-and-Stripes CSV the team uses lives next to this folder in the repo.
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "Bayesian Model" / "bars_stripes_128.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "vae_results.csv"


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def median_distance(x: torch.Tensor) -> float:
    """Median Euclidean distance between pairs of rows (for 0/1 data: sqrt of Hamming distance)."""
    x = x.double()
    d = torch.cdist(x, x)
    iu = torch.triu_indices(len(x), len(x), offset=1)
    return d[iu[0], iu[1]].median().item()


def gaussian_mmd2(x: torch.Tensor, y: torch.Tensor, sigmas: tuple) -> float:
    """Unbiased MMD^2 between sample sets x and y (any sizes) with a Gaussian kernel
    averaged over `sigmas`. ~0 when x and y come from the same distribution."""
    x, y = x.double(), y.double()

    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s * s)) for s in sigmas) / len(sigmas)

    m, n = len(x), len(y)
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    xx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))  # unbiased: skip self-pairs
    yy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return (xx + yy - 2 * kxy.mean()).item()


def ignite_mmd(x: torch.Tensor, y: torch.Tensor, var: float = 1.0):
    """ignite's MMD (what the team's other scripts use). Returns None if ignite isn't installed."""
    try:
        from ignite.metrics import MaximumMeanDiscrepancy
    except ImportError:
        return None
    n = min(len(x), len(y))  # ignite requires both sets to have the same shape
    metric = MaximumMeanDiscrepancy(var=var)
    metric.update((x[:n].float(), y[:n].float()))
    return float(metric.compute())


def bas_bit_errors(samples: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """For each sample: the fewest bit flips needed to make it a valid Bars-and-Stripes
    pattern on a rows x cols grid (every row constant = horizontal stripes, OR every
    column constant = vertical bars). 0 means the sample is already valid."""
    grid = samples.reshape(-1, rows, cols)
    ones_per_row, ones_per_col = grid.sum(dim=2), grid.sum(dim=1)
    flips_to_stripes = torch.minimum(ones_per_row, cols - ones_per_row).sum(dim=1)
    flips_to_bars = torch.minimum(ones_per_col, rows - ones_per_col).sum(dim=1)
    return torch.minimum(flips_to_stripes, flips_to_bars)


# ----------------------------------------------------------------------------------------
# One experiment (= one seed)
# ----------------------------------------------------------------------------------------
def run_one(data: torch.Tensor, args, seed: int, check_determinism: bool) -> dict:
    kwargs = dict(seed=seed, train_frac=args.train_frac, epochs=args.epochs, n_samples=args.n_samples,
                  latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                  lr=args.lr, beta=args.beta)

    print(f"\n=== seed {seed} ===")
    out = run_vae(data, verbose=args.verbose, **kwargs)
    gen, held_out, train = out["generated_samples"], out["test_data"], out["train_data"]
    n_bits = data.shape[1]

    # Baseline: uniformly random bitstrings, same count and length as the VAE's samples.
    rng = torch.Generator().manual_seed(seed + 1)
    random_bits = torch.bernoulli(torch.full((args.n_samples, n_bits), 0.5), generator=rng)

    # One bandwidth for all comparisons in this run (taken from the real held-out data).
    s0 = median_distance(held_out) or 1.0  # fall back to 1.0 if the median distance is 0 (very repetitive data)
    sigmas = (0.5 * s0, s0, 2 * s0)

    row = {
        "dataset": Path(args.data).name, "model": "VAE", "seed": seed,
        "train_frac": args.train_frac, "epochs": args.epochs, "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim, "beta": args.beta, "lr": args.lr,
        "n_bits": n_bits, "n_train": len(train), "n_test": len(held_out), "n_generated": len(gen),
        "final_train_loss": round(out["final_train_loss"], 4),
        "test_loss": round(out["test_loss"], 4),
        "mmd2_vae": gaussian_mmd2(gen, held_out, sigmas),
        "mmd2_random": gaussian_mmd2(random_bits, held_out, sigmas),
        "mmd2_train_ref": gaussian_mmd2(train, held_out, sigmas),
        "ignite_mmd_vae": ignite_mmd(gen, held_out),
        "ignite_mmd_random": ignite_mmd(random_bits, held_out),
        "validity_generated": None, "validity_heldout": None,
        "bit_errors_generated": None, "bit_errors_random": None, "deterministic": None,
    }

    if args.rows * args.cols == n_bits:  # Bars-and-Stripes-specific sanity check
        err_gen = bas_bit_errors(gen, args.rows, args.cols)
        err_held = bas_bit_errors(held_out, args.rows, args.cols)
        err_rand = bas_bit_errors(random_bits, args.rows, args.cols)
        row["validity_generated"] = (err_gen == 0).float().mean().item()
        row["validity_heldout"] = (err_held == 0).float().mean().item()
        row["bit_errors_generated"] = err_gen.float().mean().item()
        row["bit_errors_random"] = err_rand.float().mean().item()

    if check_determinism:
        again = run_vae(data, **kwargs)
        row["deterministic"] = bool(torch.equal(gen, again["generated_samples"])
                                    and out["train_loss"] == again["train_loss"])

    if args.save_samples:
        path = Path(args.save_samples)
        if len(args.seeds) > 1:  # one file per seed so runs don't overwrite each other
            path = path.with_name(f"{path.stem}_seed{seed}{path.suffix}")
        save_bitstrings(gen, str(path))
        print(f"saved generated samples -> {path}")

    return row


def print_report(row: dict) -> None:
    def f(v, spec):
        return "n/a" if v is None else format(v, spec)

    beats = row["mmd2_vae"] < row["mmd2_random"]
    print(f"  split: {row['n_train']} train / {row['n_test']} held-out | epochs {row['epochs']} | "
          f"latent {row['latent_dim']} | beta {row['beta']}")
    print(f"  loss (neg. ELBO per bitstring):  train {row['final_train_loss']:.2f} | held-out {row['test_loss']:.2f}")
    print(f"  {'':<28}{'MMD2 (gaussian)':>17}{'MMD (ignite, var=1)':>22}")
    print(f"  {'VAE generated':<28}{row['mmd2_vae']:>17.5f}{f(row['ignite_mmd_vae'], '.5f'):>22}")
    print(f"  {'random bitstrings':<28}{row['mmd2_random']:>17.5f}{f(row['ignite_mmd_random'], '.5f'):>22}")
    print(f"  {'real train rows (floor)':<28}{row['mmd2_train_ref']:>17.5f}{'':>22}")
    print(f"  VAE beats random baseline (gaussian MMD2): {'YES' if beats else 'NO'}")
    if row["validity_generated"] is not None:
        print(f"  valid Bars-and-Stripes patterns: generated {row['validity_generated']:.1%} "
              f"(held-out real data: {row['validity_heldout']:.1%})")
        print(f"  avg bits wrong vs nearest valid pattern (of {row['n_bits']}): "
              f"generated {row['bit_errors_generated']:.1f} | random {row['bit_errors_random']:.1f} | real 0.0")
    if row["deterministic"] is not None:
        print(f"  reproducible with same seed: {'YES (identical output)' if row['deterministic'] else 'NO - outputs differ!'}")


def append_results(row: dict, path: Path) -> None:
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with a 'bitstring' column")
    ap.add_argument("--seeds", type=int, nargs="+", default=[50], help="one run per seed (default: 50)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=1000, help="bitstrings to generate")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=2.0, help="KL weight (1.0 = standard VAE)")
    ap.add_argument("--rows", type=int, default=8, help="grid rows, for the Bars-and-Stripes validity check")
    ap.add_argument("--cols", type=int, default=16, help="grid columns (rows*cols must equal bitstring length)")
    ap.add_argument("--save-samples", default=None, help="write generated bitstrings to this CSV")
    ap.add_argument("--results", default=str(RESULTS_CSV), help="CSV that results are appended to")
    ap.add_argument("--skip-determinism-check", action="store_true", help="skip the (2x slower) same-seed check")
    ap.add_argument("--verbose", action="store_true", help="print training loss during training")
    args = ap.parse_args()

    if not Path(args.data).exists():
        ap.error(f"data file not found: {args.data}  (pass --data path/to/bitstrings.csv)")
    if not 0 < args.train_frac < 1:
        ap.error("--train-frac must be strictly between 0 and 1 (a held-out set is needed for MMD)")

    data = load_bitstrings(args.data)
    print(f"loaded {args.data}: {data.shape[0]} bitstrings x {data.shape[1]} bits")

    rows = []
    for i, seed in enumerate(args.seeds):
        row = run_one(data, args, seed, check_determinism=not args.skip_determinism_check and i == 0)
        print_report(row)
        append_results(row, Path(args.results))
        rows.append(row)

    if len(rows) > 1:
        wins = sum(r["mmd2_vae"] < r["mmd2_random"] for r in rows)
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        print(f"\n=== summary over {len(rows)} seeds ===")
        print(f"  VAE beat the random baseline in {wins}/{len(rows)} runs")
        print(f"  mean MMD2: VAE {mean('mmd2_vae'):.5f} | random {mean('mmd2_random'):.5f} | "
              f"real-train floor {mean('mmd2_train_ref'):.5f}")
    print(f"\nresults appended to {args.results}")


if __name__ == "__main__":
    main()
