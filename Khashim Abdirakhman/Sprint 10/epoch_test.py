"""epoch test: where does the QCBM stop improving on the 12-bit qubit set? -> results/epoch_test.json

python epoch_test.py
python epoch_test.py --max-epochs 10000 --patience 1000 --max-minutes 25
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mmd import mmd_samples
from qcbm import Config, fit, sample_bits

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=1000) # stop if the train MMD hasnt improved in this many epochs
    ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    data = torch.tensor(pd.read_csv(TEAM / "training_sequences_qubit.csv", header=None).values, dtype=torch.float64)
    train_x, test_x = train_test_split(data, train_size=args.epsilon, random_state=args.seed) # same split as the framework

    cfg = Config(seed=args.seed, steps=args.max_epochs, patience=args.patience, log_every=1000,
                 max_seconds=args.max_minutes * 60)
    model = fit(train_x, cfg, verbose=True)

    p = model().detach()
    gen = sample_bits(p, len(test_x), torch.Generator().manual_seed(args.seed))
    out = {
        "dataset": "qubit (12 bits)", "n_train": len(train_x), "epsilon": args.epsilon, "seed": args.seed,
        "epochs_run": len(model.history), "plateau_epoch": model.best["step"], "stop": model.stop,
        "best_train_mmd": model.best["loss"], "train_seconds": round(model.train_seconds, 1),
        "test_mmd_samples": mmd_samples(gen, test_x).item(),
        "train_mmd_every_1000": model.history[999::1000],
    }
    (HERE / "results" / "epoch_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "train_mmd_every_1000"}, indent=2))


if __name__ == "__main__":
    main()
