"""Seeds + sample size sweep (sprint 8) -> results/stability/STABILITY.md

python stability.py (100 runs, Take about 1-1.5 hrs)
python stability.py --quick
python stability.py --datasets bas markov --seeds 3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import data_generator as dg
from qcbm import Config, run_name, train

# 2 questions:
# 1. are the sprint 7 results real or just luck from 1 seed? -> run every setting w/ 5 seeds, report mean +- std
# 2. at what # of samples does plain counting start beating the qcbm? -> sizes 50..1000
#
# seed s = new training samples AND new starting angles, so the spread includes both
#
#runs already done are loaded from their json, so if it crashes / laptop sleeps just run it again and it continues

HERE = Path(__file__).resolve().parent
SIZES = [50, 100, 250, 500, 1000]
MODELS = ["qcbm", "counting", "smoothed"]
LOWER_BETTER = {"tvd", "mmd", "kl", "test_mmd_samples"}


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def run_all(args) -> dict:
    out = Path(args.out)
    results = {} # (dataset, n) -> 1 result per seed
    total, k, t0 = len(args.datasets) * len(args.sizes) * args.seeds, 0, time.time()
    for name in args.datasets:
        ds = dg.make(name)
        p_true = torch.tensor(ds.target, dtype=torch.float64)
        for n in args.sizes:
            for s in range(args.seeds):
                k += 1
                cfg = Config(dataset=name, ansatz=args.ansatz, layers=args.layers, steps=args.steps,
                             seed=s, out_dir=str(out), log_every=10 ** 9)
                path = out / f"{run_name(cfg, n)}.json"
                r = json.loads(path.read_text()) if path.exists() else None
                # only reuse an old run if it used the same settings
                if r is None or r["config"]["steps"] != args.steps or r["config"]["layers"] != args.layers:
                    x = torch.tensor(ds.sample(n, seed=s), dtype=torch.float64)
                    r = train(cfg, x, p_true, ds.meta, save=True, verbose=False)
                results.setdefault((name, n), []).append(r)
                print(f"[{k}/{total}] {name:<8} n={n:<5} seed={s} | TVD qcbm {r['qcbm']['tvd']:.3f} "
                      f"counting {r['counting']['tvd']:.3f} smoothed {r['smoothed']['tvd']:.3f} "
                      f"| {time.time() - t0:.0f}s total")
    return results


def crossover(rows) -> str:
    # rows = [(n, qcbm mean tvd, counting mean tvd)]
    qcbm_wins = [q < c for _, q, c in rows]
    if all(qcbm_wins):
        return f"QCBM has lower mean TVD than counting at every size tested (up to {rows[-1][0]})."
    if not any(qcbm_wins):
        return "Counting has lower mean TVD than the QCBM at every size tested."
    # smallest n where counting wins at n and every bigger n
    for i in range(len(rows)):
        if not any(qcbm_wins[i:]):
            return f"Counting catches up at n = {rows[i][0]} and stays better for every bigger size."
    return "Mixed: no clean crossover, winner changes back and forth between sizes (look at the spread)."


def report(results, args) -> str:
    md = ["# QCBM stability + sample size sweep", "",
          f"circuit: {args.ansatz} ({args.layers} layers), {args.steps} steps max, {args.seeds} seeds per setting.",
          "Each seed = new training samples AND new starting angles.",
          "Numbers are mean ± std over seeds. All metrics are vs the TRUE distribution.",
          "\"QCBM wins\" = in how many seeds the QCBM beat plain counting on that metric.",
          "smoothed = counting + 0.5 pseudo-count on every bitstring.", ""]
    md += ["## Where does counting catch up? (mean TVD)", ""]
    for name in args.datasets:
        rows = []
        for n in args.sizes:
            rs = results[(name, n)]
            rows.append((n, mean_std([r["qcbm"]["tvd"] for r in rs])[0], mean_std([r["counting"]["tvd"] for r in rs])[0]))
        md.append(f"- **{name}**: {crossover(rows)}")
    md.append("")

    for name in args.datasets:
        metrics = ["tvd", "mmd", "kl", "coverage"] + (["validity"] if name == "bas" else [])
        md += [f"## {name}", "", "| n | metric | QCBM | counting | smoothed | QCBM wins |", "|---|---|---|---|---|---|"]
        for n in args.sizes:
            rs = results[(name, n)]
            for key in metrics:
                cells = []
                for m in MODELS:
                    mu, sd = mean_std([r[m][key] for r in rs])
                    cells.append(f"{mu:.4f} ± {sd:.4f}")
                if key in LOWER_BETTER:
                    wins = sum(r["qcbm"][key] < r["counting"][key] for r in rs)
                else:
                    wins = sum(r["qcbm"][key] > r["counting"][key] for r in rs)
                md.append(f"| {n} | {key} | " + " | ".join(cells) + f" | {wins}/{len(rs)} |")
        md.append("")

    secs = [r["train_seconds"] for rs in results.values() for r in rs]
    md += ["## Run details", "", f"{len(secs)} runs, {sum(secs) / 60:.1f} min of training total, "
           f"{np.mean(secs):.1f}s per run on average."]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=dg.NAMES, choices=dg.NAMES)
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--seeds", type=int, default=5, help="how many seeds (0..seeds-1)")
    ap.add_argument("--ansatz", default="strong", choices=["strong", "hea", "iqp"])
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "results" / "stability"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        # separate folder so quick runs dont mix w/ the real ones
        args.seeds, args.sizes, args.steps = 2, [50, 1000], 40
        args.out = str(HERE / "results" / "stability_quick")

    results = run_all(args)
    md = report(results, args)
    out_md = Path(args.out) / "STABILITY.md"
    out_md.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"saved -> {out_md}")


if __name__ == "__main__":
    main()
