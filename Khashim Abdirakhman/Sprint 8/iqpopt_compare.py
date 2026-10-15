"""IQPopt vs my pennylane IQP circuit (sprint 8) -> results/iqpopt/IQPOPT.md

pip install git+https://github.com/gdeluca1/iqpopt.git networkx
python iqpopt_compare.py
python iqpopt_compare.py --scale 4 5
python iqpopt_compare.py --quick
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

import iqpopt as iqp
import iqpopt.gen_qml as gen
from iqpopt.utils import local_gates

import data_generator as dg
from mmd import DEFAULT_SIGMAS, kernel_matrix, mmd_samples
from qcbm import Config, build_circuit, coverage, distribution_metrics, run_name, sample_bits, train

# what this checks:
# 1. same circuit 2 ways - my pennylane "iqp" (exact MMD, torch) vs Sponsor's iqpopt (estimated MMD, jax), only the loss + optimizer are different
# 2. equivalence - my trained params / 2 put into iqpopt -> probs should match ~1e-12, if not the comparison means nothing
# 3. kernel - iqpopt's MMD estimate vs my mmd_samples on the same test set, should be close
# 4. scale - iqpopt never builds the 2^n state for training, so try 4x4 (16 qubits) and 5x5 (25 qubits) bas
#
# iqpopt gate = exp(-i θ X..X), mine RZ(θ) / IsingZZ(θ) = exp(-i θ/2 Z..Z) -> iqpopt θ = my θ / 2
# gate order is the same: local_gates(n, 2) = singles 0..n-1 then pairs (0,1),(0,2).. same as my `pairs`

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────── IQPopt helpers ────────────────────────────────────────
def make_iqpopt(n_qubits: int):
    gates = local_gates(n_qubits, 2)
    # default.qubit bc iqpopt's default lightning.qubit is an extra install
    return iqp.IqpSimulator(n_qubits, gates, device="default.qubit"), gates


def train_iqpopt(train_x: np.ndarray, n_qubits: int, args, seed: int):
    circuit, gates = make_iqpopt(n_qubits)
    rng = np.random.default_rng(seed)
    params0 = jnp.array(rng.normal(0, 1 / np.sqrt(n_qubits), len(gates))) # same init as the iqpopt readme
    loss_kwargs = {
        "params": params0,
        "iqp_circuit": circuit,
        "ground_truth": jnp.array(train_x, dtype=jnp.float64),
        "sigma": list(args.sigmas),
        "n_ops": args.n_ops,
        "n_samples": args.n_mc,
    }
    trainer = iqp.Trainer("Adam", gen.mmd_loss_iqp, stepsize=args.iqpopt_lr)
    t0 = time.time()
    trainer.train(n_iters=args.steps, loss_kwargs=loss_kwargs, random_state=seed)
    secs = time.time() - t0 # includes jax compile time on the 1st step
    return circuit, params0, trainer.final_params, np.asarray(trainer.losses), secs


def iqpopt_test_mmd(circuit, params, test_x: np.ndarray, args, seed: int) -> float:
    # iqpopt's MMD^2 estimate, works at any # of qubits
    return float(gen.mmd_loss_iqp(jnp.asarray(params), circuit, jnp.array(test_x, dtype=jnp.float64),
                                  list(args.sigmas), args.n_ops, args.n_mc, jax.random.PRNGKey(seed)))


def iqpopt_probs(circuit, params) -> torch.Tensor:
    # full 2^n prob vector, only for small n
    return torch.tensor(np.asarray(circuit.probs(jnp.asarray(params))), dtype=torch.float64)


# ─────────────────────────────────────────────── 3x3 comparison ────────────────────────────────────────
def compare_small(args, out: Path) -> dict:
    ds = dg.bars_and_stripes(3)
    n = ds.n_bits
    x_np = ds.sample(args.n_train, seed=args.seed)
    test_np = ds.sample(args.test_shots, seed=10_000 + args.seed)
    train_x = torch.tensor(x_np, dtype=torch.float64)
    test_x = torch.tensor(test_np, dtype=torch.float64)
    p_true = torch.tensor(ds.target, dtype=torch.float64)
    support = torch.tensor(ds.support)
    sigmas = tuple(args.sigmas)
    K = kernel_matrix(n, sigmas)

    print("\n── pennylane iqp ──")
    cfg = Config(dataset="bas", ansatz="iqp", steps=args.steps, seed=args.seed, sigmas=sigmas,
                 test_shots=args.test_shots, out_dir=str(out / "pennylane"), log_every=100)
    r_pl = train(cfg, train_x, p_true, ds.meta)
    theta_pl = torch.load(out / "pennylane" / f"{run_name(cfg, args.n_train)}_params.pt").double()

    print("\n── iqpopt iqp ──")
    circuit, p0, params, losses, secs = train_iqpopt(x_np, n, args, args.seed)
    q_iqp = iqpopt_probs(circuit, params)
    m_iqp = distribution_metrics(p_true, q_iqp, K, support)
    g = torch.Generator().manual_seed(args.seed)
    m_iqp["coverage"] = coverage(sample_bits(q_iqp, args.n_train, g), support)
    m_iqp["test_mmd_samples"] = mmd_samples(sample_bits(q_iqp, args.test_shots, g), test_x, sigmas).item()

    # equivalence check
    with torch.no_grad():
        q_pl = build_circuit("iqp", n)(theta_pl)
    q_pl_in_iqpopt = iqpopt_probs(circuit, theta_pl.numpy() / 2)
    max_diff = (q_pl - q_pl_in_iqpopt).abs().max().item()

    # kernel check
    m_iqp["test_mmd_iqpopt_est"] = iqpopt_test_mmd(circuit, params, test_np, args, args.seed)
    pl_est = iqpopt_test_mmd(circuit, theta_pl.numpy() / 2, test_np, args, args.seed)

    return {
        "n_qubits": n, "n_params": len(circuit.gates), "max_prob_diff_equivalence": max_diff,
        "pennylane": {**r_pl["qcbm"], "test_mmd_iqpopt_est": pl_est, "seconds": r_pl["train_seconds"],
                      "steps": r_pl["steps_run"]},
        "iqpopt": {**m_iqp, "seconds": round(secs, 1), "steps": len(losses),
                   "final_train_loss": float(losses[-1]) if len(losses) else None},
        "counting": r_pl["counting"], "smoothed": r_pl["smoothed"],
    }


# ─────────────────────────────────────────────── Bigger grids ──────────────────────────────────────────
def scale_run(size: int, args) -> dict:
    n = size * size
    pats = dg.bas_patterns(size)
    x_np = dg.bas_sample(size, args.n_train, seed=args.seed)
    test_np = dg.bas_sample(size, args.test_shots, seed=10_000 + args.seed)
    print(f"\n── iqpopt bas {size}x{size} ({n} qubits) ──")
    circuit, p0, params, losses, secs = train_iqpopt(x_np, n, args, args.seed)

    row = {
        "size": size, "n_qubits": n, "n_params": len(circuit.gates), "seconds": round(secs, 1),
        "steps": len(losses), "valid_patterns": len(pats), "random_validity": len(pats) / 2 ** n,
        "test_mmd_init": iqpopt_test_mmd(circuit, p0, test_np, args, args.seed),
        "test_mmd_trained": iqpopt_test_mmd(circuit, params, test_np, args, args.seed),
        #real train vs real test -> what a perfect model would get (~0)
        "test_mmd_reference": mmd_samples(torch.tensor(x_np, dtype=torch.float64),
                                          torch.tensor(test_np, dtype=torch.float64), tuple(args.sigmas)).item(),
    }
    if n <= args.max_exact_qubits:
        q = iqpopt_probs(circuit, params)
        support = torch.tensor(dg.bits_to_index(pats))
        row["validity"] = q[support].sum().item()
        g = torch.Generator().manual_seed(args.seed)
        row["coverage"] = coverage(sample_bits(q, args.n_train, g), support)
    return row


# ─────────────────────────────────────────────── Report ────────────────────────────────────────────────
def f4(x):
    return "—" if x is None else f"{x:.4f}"


def report(small: dict, big: list, args) -> str:
    md = ["# IQPopt vs PennyLane IQP", "",
          f"Bars and Stripes, {args.n_train} training samples, {args.test_shots} test samples, seed {args.seed}, "
          f"sigmas {list(args.sigmas)}.",
          f"IQPopt: Adam lr {args.iqpopt_lr}, n_ops {args.n_ops}, n_samples {args.n_mc}. "
          "PennyLane: Adam lr 0.05 w/ cosine schedule + early stopping (same as qcbm.py).",
          "Same circuit family for both: 1 parameter per qubit + 1 per pair.", ""]
    if small:
        diff = small["max_prob_diff_equivalence"]
        ok = "PASS" if diff < 1e-8 else "FAIL - circuits are NOT the same, don't trust the comparison"
        keys = [("MMD", "mmd"), ("TVD", "tvd"), ("KL", "kl"), ("Fidelity", "fidelity"), ("Validity", "validity"),
                ("Coverage", "coverage"), ("Test MMD (mine)", "test_mmd_samples"),
                ("Test MMD (iqpopt est.)", "test_mmd_iqpopt_est"), ("steps", "steps"), ("seconds", "seconds")]
        md += [f"## 3x3 ({small['n_qubits']} qubits, {small['n_params']} parameters)", "",
               f"Equivalence check (my params / 2 in IQPopt): max prob difference = {diff:.2e} -> {ok}", "",
               "| model | " + " | ".join(k for k, _ in keys) + " |", "|" + "---|" * (len(keys) + 1)]
        for name in ["pennylane", "iqpopt", "counting", "smoothed"]:
            m = small[name]
            cells = [str(m[k]) if k == "steps" and k in m else f4(m.get(k)) for _, k in keys]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md += ["", "Test MMD (mine) and Test MMD (iqpopt est.) should be close for the same model. "
               "If they are far apart, the kernels or sigmas don't match.", ""]
    if big:
        md += ["## Bigger grids (IQPopt only)", "",
               "test MMD = IQPopt's estimate vs held-out samples. reference = real train set vs real test set "
               "(what a perfect model would get, ~0). Validity/coverage only where exact probs are still possible.", "",
               "| grid | qubits | params | test MMD init | test MMD trained | reference | validity | random validity | coverage | seconds |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in big:
            md.append(f"| {r['size']}x{r['size']} | {r['n_qubits']} | {r['n_params']} | {f4(r['test_mmd_init'])} | "
                      f"{f4(r['test_mmd_trained'])} | {f4(r['test_mmd_reference'])} | {f4(r.get('validity'))} | "
                      f"{r['random_validity']:.2e} | {f4(r.get('coverage'))} | {r['seconds']} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--test-shots", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--iqpopt-lr", type=float, default=0.02, help="iqpopt θ is half of mine, so 0.02 ~ 0.04 in my units")
    ap.add_argument("--n-ops", type=int, default=1000, help="iqpopt MMD estimate: # of Z operators")
    ap.add_argument("--n-mc", type=int, default=1000, help="iqpopt MMD estimate: # of monte carlo samples")
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, nargs="*", default=[], help="extra bas grid sizes, e.g. 4 5")
    ap.add_argument("--max-exact-qubits", type=int, default=16)
    ap.add_argument("--skip-small", action="store_true", help="only do the --scale runs")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.n_ops, args.n_mc, args.test_shots = 30, 200, 200, 500

    out = HERE / "results" / ("iqpopt_quick" if args.quick else "iqpopt")
    out.mkdir(parents=True, exist_ok=True)
    small = None if args.skip_small else compare_small(args, out)
    big = [scale_run(s, args) for s in args.scale]

    md = report(small, big, args)
    (out / "IQPOPT.md").write_text(md, encoding="utf-8")
    (out / "iqpopt_compare.json").write_text(json.dumps({"args": vars(args), "small": small, "scale": big}, indent=2))
    print("\n" + md)
    print(f"saved -> {out / 'IQPOPT.md'}")


if __name__ == "__main__":
    main()
