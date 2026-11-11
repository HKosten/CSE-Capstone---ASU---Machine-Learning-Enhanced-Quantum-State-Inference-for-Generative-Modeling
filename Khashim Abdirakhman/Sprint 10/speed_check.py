"""QCBM speed + memory check at wider sizes (product-kernel loss, no 2^n x 2^n matrix)

python speed_check.py
python speed_check.py --bits 12 14 16 18 --steps 10
"""

import argparse
import json
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from qcbm import Config, fit

HERE = Path(__file__).resolve().parent


class PeakRam:
    # polls this process every 50ms, bc torch doesnt report cpu memory itself
    def __init__(self):
        self.proc, self.peak, self.run = psutil.Process(), 0, True
        self.t = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while self.run:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.run = False
        self.t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, nargs="+", default=[12, 14, 16, 18])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rows", type=int, default=300)
    args = ap.parse_args()

    data = pd.read_csv(HERE.parent.parent / "Team" / "training_sequences_DNA.csv", header=None).values[:args.rows]
    out = []
    fit(torch.tensor(data[:, :8], dtype=torch.float64), Config(seed=33, steps=3), verbose=False) # warm up, 1st call is slow
    for n in args.bits:
        x = torch.tensor(data[:, :n], dtype=torch.float64)
        with PeakRam() as ram:
            t0 = time.time()
            m = fit(x, Config(seed=33, steps=args.steps), verbose=False)
            secs = time.time() - t0
        row = {"bits": n, "steps": len(m.history), "seconds_per_step": round(secs / len(m.history), 3),
               "peak_ram_gb": round(ram.peak / 2**30, 2), "params": m.theta.numel(),
               "old_kernel_matrix_gb": round((2 ** n) ** 2 * 8 / 2**30, 2)}
        print(row, flush=True)
        out.append(row)
    (HERE / "results" / "speed_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
