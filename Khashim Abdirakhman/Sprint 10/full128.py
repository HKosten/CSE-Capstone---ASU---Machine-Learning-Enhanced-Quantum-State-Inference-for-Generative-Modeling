"""one IQPopt circuit on all 128 bits of the sin set (no windows), compared with the window model -> results/full128_sin.json

python full128.py
python full128.py --minutes 1

training works at any width (iqpopt never builds the state), sampling does not (~26 bits max), so it is judged with iqpopt's own
MMD estimate and with <Z0 Z127> style expectations instead of samples. window samples come from results/samples.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import iqp_windows as iw
from mmd import DEFAULT_SIGMAS, mmd_samples

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent.parent / "Team"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--n-ops", type=int, default=1000)
    args = ap.parse_args()

    jnp, iqp, gen, nn_gates, _ = iw.load_iqpopt()
    import jax

    data = pd.read_csv(TEAM / "sin_bitstrings.csv", header=None).values
    train, test = train_test_split(data, train_size=0.5, random_state=33) # same split as the window run
    n = data.shape[1]
    gates = nn_gates(n)
    sim = iqp.IqpSimulator(n, gates, device="default.qubit")

    t0 = time.time()
    params, losses = iw.train_iqp(sim, train, len(gates), 10**6, 33, seconds=args.minutes * 60) # same settings as the windows
    secs = time.time() - t0
    params0 = np.random.default_rng(33).normal(0, 1 / np.sqrt(n), len(gates))
    print(f"trained {len(losses)} steps in {secs:.0f}s, loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)

    truth = jnp.array(test, dtype=jnp.float64)
    sig = list(DEFAULT_SIGMAS)

    def est(p, seed):
        return float(gen.mmd_loss_iqp(jnp.asarray(p), sim, truth, sig, args.n_ops, 1000, jax.random.PRNGKey(seed)))
    trained = [est(params, s) for s in range(5)] # mmd^2 estimate, 5 different random keys
    init = [est(params0, s) for s in range(5)]

    win = np.load(HERE / "results" / "samples" / "sin_iqpwindows_eps0.5_seed33.npy").astype(np.float64)
    win_iqpopt = float(np.mean([float(gen.mmd_loss_samples(truth, jnp.asarray(win), s)) for s in sig])) # iqpopt's own sample estimator
    win_mine = mmd_samples(torch.tensor(win), torch.tensor(test, dtype=torch.float64)).item() # shared mmd.py, same kernels
    coin = np.random.default_rng(0).integers(0, 2, win.shape).astype(np.float64)
    coin_mine = mmd_samples(torch.tensor(coin), torch.tensor(test, dtype=torch.float64)).item()

    def corr(x):
        return float(np.corrcoef(x[:, 0], x[:, n - 1])[0, 1])
    ops = np.zeros((3, n))
    ops[0, 0], ops[1, n - 1], ops[2, [0, n - 1]] = 1, 1, 1 # <s0>, <s127>, <s0 s127> with s = (-1)^bit
    try:
        ev = np.asarray(sim.op_expval(jnp.asarray(params), jnp.asarray(ops), 20000, jax.random.PRNGKey(0))[0])
        circuit_corr = float((ev[2] - ev[0] * ev[1]) / np.sqrt((1 - ev[0] ** 2) * (1 - ev[1] ** 2)))
    except Exception as e: # not worth losing the rest over
        circuit_corr, ev = None, str(e)[:200]

    out = {"bits": n, "gates": "nearest neighbour", "n_params": len(gates), "train_rows": len(train), "test_rows": len(test),
           "train_seconds": round(secs, 1), "steps": len(losses), "loss_first": losses[0], "loss_last": losses[-1],
           "full_circuit_iqpopt_mmd2": {"mean": float(np.mean(trained)), "std": float(np.std(trained))},
           "full_circuit_untrained_iqpopt_mmd2": {"mean": float(np.mean(init)), "std": float(np.std(init))},
           "windows_iqpopt_sample_mmd2": win_iqpopt, "windows_shared_mmd2": win_mine, "coin_flips_shared_mmd2": coin_mine,
           "corr_bit0_bit127": {"real_all_rows": corr(data), "real_train": corr(train), "windows_samples": corr(win),
                                "full_circuit_op_expval": circuit_corr}}
    (HERE / "results" / "full128_sin.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
