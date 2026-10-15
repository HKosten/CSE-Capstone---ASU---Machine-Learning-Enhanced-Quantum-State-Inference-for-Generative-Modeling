"""Sprint 7 grid again w/ the sprint 8 fixes -> results/SUMMARY.md

python run_experiments.py (Take about 10-20 mins)
python run_experiments.py --quick
"""

import argparse
import subprocess
import sys
from pathlib import Path

from qcbm import Config, train

# steps:
# 1. Make every dataset w/ 1000 samples and w/ 100 samples
# 2. Train each circuit type (strong, hea, iqp) on the 1000 sample data
# 3. Train strong on the 100 sample data - few samples is where a model should beat plain counting (if it ever does)
# 4. Each run saved as json + one markdown summary table
#
# changes from sprint 7:
# - FIX paths are relative to this file, sprint 7 crashed if you ran it from another folder
# - smoothed baseline rows added
# - test MMD column added
# - --data-seed / --seed flags. defaults (0 / 42) = same data + same start as sprint 7

HERE = Path(__file__).resolve().parent
DATASETS = ["bas", "markov", "gaussian", "quantum"]
ANSATZES = ["strong", "hea", "iqp"]
KEYS = [("MMD", "mmd"), ("TVD", "tvd"), ("KL", "kl"), ("Fidelity", "fidelity"),
        ("Validity", "validity"), ("Coverage", "coverage"), ("Test MMD", "test_mmd_samples")]


def generate(n_samples: int, out: Path, seed: int):
    subprocess.run([sys.executable, str(HERE / "data_generator.py"), "--all", "--n-samples", str(n_samples),
                    "--seed", str(seed), "--out", str(out)], check=True)


def fmt(x):
    return "—" if x is None else f"{x:.4f}"


def table(rows, keys=KEYS):
    # rows -> markdown table
    head = "| dataset | model | " + " | ".join(k for k, _ in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [head, sep]
    for ds, model, m in rows:
        lines.append(f"| {ds} | {model} | " + " | ".join(fmt(m.get(k)) for _, k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    steps = 40 if args.quick else 400

    d1000, d100 = HERE / "data" / "n1000", HERE / "data" / "n100"
    generate(1000, d1000, args.data_seed)
    generate(100, d100, args.data_seed)

    main_rows, small_rows, info = [], [], []

    # 1000 samples, every circuit type
    for ds in DATASETS:
        r = None
        for az in ANSATZES:
            r = train(Config(dataset=ds, data_dir=str(d1000), ansatz=az, steps=steps, seed=args.seed,
                             out_dir=str(HERE / "results" / "n1000"), log_every=100))
            main_rows.append((ds, f"QCBM-{az}", r["qcbm"]))
            info.append((ds, az, r["n_parameters"], r["steps_run"], r["train_seconds"]))
        # same for every ansatz, just keep the last one
        main_rows += [(ds, "counting", r["counting"]), (ds, "smoothed", r["smoothed"])]

    # 100 samples, strong only
    for ds in DATASETS:
        r = train(Config(dataset=ds, data_dir=str(d100), ansatz="strong", steps=steps, seed=args.seed,
                         out_dir=str(HERE / "results" / "n100"), log_every=100))
        small_rows += [(ds, "QCBM-strong", r["qcbm"]), (ds, "counting", r["counting"]),
                       (ds, "smoothed", r["smoothed"])]

    md = ["# QCBM experiment results (sprint 8 rerun)", "",
          "All metrics compare the model with the TRUE distribution, which the model never saw.",
          "MMD, TVD, KL, Test MMD: lower is better. Fidelity, Validity, Coverage: higher is better.",
          "Validity only applies to Bars and Stripes (the only dataset with invalid bitstrings).",
          "Coverage: every model gets the same number of draws as there were training samples.",
          "smoothed = counting + 0.5 pseudo-count on every bitstring.", "",
          f"data seed {args.data_seed}, init seed {args.seed}", "",
          "## Trained on 1000 samples", "", table(main_rows), "",
          "## Trained on 100 samples", "", table(small_rows), "",
          "## Run details", "", "| dataset | ansatz | parameters | steps | seconds |", "|---|---|---|---|---|"]
    md += [f"| {d} | {a} | {p} | {s} | {t} |" for d, a, p, s, t in info]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
