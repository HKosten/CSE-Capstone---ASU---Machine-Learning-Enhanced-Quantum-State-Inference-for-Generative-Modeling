"""does my model return the same thing as the team models? calls all 4 the way evaluation_framework.py does -> results/consistency_check.json

python consistency_check.py
python consistency_check.py --samples 137 --epochs 2

same split + same call (model(**settings, training_data=train_tensor)) as Team/evaluation_framework.py, 2 epochs so it is quick.
checks type, dtype, shape == (num_samples, n_bits), only 0/1. also checks the files the last Main.py run saved.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"
sys.dont_write_bytecode = True # dont leave __pycache__ in Team
sys.path.insert(0, str(HERE))
sys.path.append(str(TEAM)) # mine first, so qcbm_wrapper / mmd are the sprint 10 ones

import LSTM
import qcbm_wrapper as qw
import transformer_wrapper as tw
import vae_wrapper as vw

MODELS = {"LSTM": LSTM.LSTM, "Transformer": tw.transformer_function, "VAE": vw.vae_function, "QCBM (mine)": qw.qcbm_function}
DATA = {"Qubit (12 bits)": "training_sequences_qubit.csv", "DNA (100 bits)": "training_sequences_DNA.csv"}


def describe(out, n_samples, n_bits) -> dict:
    t = torch.as_tensor(out)
    return {"type": type(out).__name__, "dtype": str(t.dtype), "shape": list(t.shape),
            "shape_ok": list(t.shape) == [n_samples, n_bits], "only_0_1": bool(((t == 0) | (t == 1)).all()),
            "is_tensor": isinstance(out, torch.Tensor)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=137) # odd number on purpose, so a hard-coded 1000 or len(train) shows up
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    results = {}
    for dname, csv in DATA.items():
        data = torch.tensor(pd.read_csv(TEAM / csv, header=None).values, dtype=torch.long)
        train, _ = train_test_split(data, train_size=0.3, random_state=33)
        settings = {"training_data": train, "num_epochs": args.epochs, "seed": 33, "epsilon": 0.3,
                    "num_samples": args.samples, "show_output": False}
        for mname, fn in MODELS.items():
            t0 = time.time()
            try:
                row = describe(fn(**settings), args.samples, data.shape[1])
            except Exception as e:
                row = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            row["seconds"] = round(time.time() - t0, 1)
            results[f"{dname} | {mname}"] = row
            print(f"{dname:<18}{mname:<14}" + ("  ".join(f"{k}={v}" for k, v in row.items())), flush=True)

    saved = {}
    for f in sorted((TEAM / "Generated Data").glob("*_generated_data_*")):
        d = pd.read_csv(f, header=None).values
        saved[f.name.split("_generated")[0]] = {"shape": list(d.shape), "only_0_1": bool(((d == 0) | (d == 1)).all())}
    results["saved_by_last_Main_run"] = saved
    print("\nfiles saved by the last Main.py run (rows x bits):")
    for k, v in saved.items():
        print(f"  {k:<26}{v['shape']}  only 0/1: {v['only_0_1']}")
    (HERE / "results" / "consistency_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
