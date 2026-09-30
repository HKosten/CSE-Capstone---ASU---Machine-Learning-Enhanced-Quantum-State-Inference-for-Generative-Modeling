"""Runs everything & Writes results / SUMMARY.md

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

DATASETS = ["bas", "markov", "gaussian", "quantum"]
ANSATZES = ["strong", "hea", "iqp"]


def generate(n_samples: int, out: str):
    subprocess.run([sys.executable, "data_generator.py", "--all", "--n-samples", str(n_samples),
                    "--out", out], check=True)


def fmt(x, key):
    return "—" if x is None else f"{x:.4f}"


def table(rows, keys):
    # rows -> markdown table
    head = "| dataset | model | " + " | ".join(k for k, _ in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [head, sep]
    for ds, model, m in rows:
        lines.append(f"| {ds} | {model} | " + " | ".join(fmt(m.get(k), k) for _, k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps = 40 if args.quick else 400

    generate(1000, "data/n1000")
    generate(100, "data/n100")

    keys = [("MMD", "mmd"), ("TVD", "tvd"), ("KL", "kl"), ("Fidelity", "fidelity"),
            ("Validity", "validity"), ("Coverage", "coverage")]
    main_rows, small_rows, info = [], [], []

    # 1000 samples, every circuit type
    for ds in DATASETS:
        base = None
        for az in ANSATZES:
            r = train(Config(dataset=ds, data_dir="data/n1000", ansatz=az, steps=steps,
                             out_dir="results/n1000", log_every=100))
            main_rows.append((ds, f"QCBM-{az}", r["qcbm"]))
            info.append((ds, az, r["n_parameters"], r["steps_run"], r["train_seconds"]))
            base = r["counting_baseline"]      # same for every ansatz, just keep the last one
        main_rows.append((ds, "counting", base))

    # 100 samples, strong only
    for ds in DATASETS:
        r = train(Config(dataset=ds, data_dir="data/n100", ansatz="strong", steps=steps,
                         out_dir="results/n100", log_every=100))
        small_rows += [(ds, "QCBM-strong", r["qcbm"]), (ds, "counting", r["counting_baseline"])]

    md = ["# QCBM experiment results", "",
          "All metrics compare the model with the TRUE distribution, which the model never saw.",
          "MMD, TVD, KL: lower is better. Fidelity, Validity, Coverage: higher is better.",
          "Validity only applies to Bars and Stripes (the only dataset with invalid bitstrings).", "",
          "## Trained on 1000 samples", "", table(main_rows, keys), "",
          "## Trained on 100 samples", "", table(small_rows, keys), "",
          "## Run details", "", "| dataset | ansatz | parameters | steps | seconds |", "|---|---|---|---|---|"]
    md += [f"| {d} | {a} | {p} | {s} | {t} |" for d, a, p, s, t in info]
    Path("results").mkdir(exist_ok=True)
    Path("results/SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
