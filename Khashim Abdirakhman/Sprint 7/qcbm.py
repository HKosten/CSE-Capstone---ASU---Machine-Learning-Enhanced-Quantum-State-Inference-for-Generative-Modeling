"""QCBM trained w/ MMD

python data_generator.py --all --out data
python qcbm.py --dataset bas --ansatz strong --layers 10 --steps 400
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

import data_generator as dg
from mmd import DEFAULT_SIGMAS, empirical_distribution, kernel_matrix, mmd_exact, mmd_samples

# QCBM = quantum circuit used as a generative model measure the circuit -> get bitstrings, prob of each one = |amplitude|^2 (born rule)
# We tune the angles so the circuit's dist. matches the data
#
# changes from v1 (sept 24):
# - trains ONLY on samples from the csv, then gets graded vs the true dist. it never saw
#   (this is the actual project goal - recover the real dist. from measured bitstrings)
# - 3 circuit options: strong (what v1 used), hea, iqp (same kind of circuit as iqpopt)
# - works w/ any dataset from data_generator.py
# - compares against just counting frequencies on the same samples
# - more metrics, lr schedule, early stopping, saves results to json (no plots)


# ─────────────────────────────────────────────── Config ────────────────────────────────────────────────
@dataclass
class Config:
    dataset: str = "bas"
    data_dir: str = "data"
    ansatz: str = "strong" # strong | hea | iqp
    layers: int = 10 # iqp ignores this, 1 block is already the whole iqp circuit
    steps: int = 400
    lr: float = 0.05
    sigmas: tuple = DEFAULT_SIGMAS
    patience: int = 80 # stop early if loss hasnt improved in this many steps
    eval_shots: int = 2000 # how many samples to draw from the model for testing
    seed: int = 42
    log_every: int = 50
    out_dir: str = "results"


# ─────────────────────────────────────────────── Circuits ──────────────────────────────────────────────
def n_params(ansatz: str, n_qubits: int, layers: int) -> tuple:
    if ansatz == "strong":
        return qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_qubits)
    if ansatz == "hea":
        return (layers, n_qubits, 2) # RY + RZ angle per qubit per layer
    if ansatz == "iqp":
        return (n_qubits + n_qubits * (n_qubits - 1) // 2,) # 1 Z per qubit + 1 ZZ per pair
    raise ValueError(f"unknown ansatz {ansatz}")


def build_circuit(ansatz: str, n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))
    pairs = [(i, j) for i in wires for j in wires if i < j]

    # backprop = fastest for small simulated circuits
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(theta):
        if ansatz == "strong":
            qml.StronglyEntanglingLayers(theta, wires=wires)
        elif ansatz == "hea":
            # hardware efficient: rotations on each qubit then a ring of CNOTs
            for layer in theta:
                for w in wires:
                    qml.RY(layer[w, 0], wires=w)
                    qml.RZ(layer[w, 1], wires=w)
                for w in wires:
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
        elif ansatz == "iqp":
            # iqp: H on everything -> Z and ZZ phases -> H again
            # (H exp(iθZ) H = exp(iθX) so this is the same as the exp(iθX..) gates in iqpopt)
            for w in wires:
                qml.Hadamard(wires=w)
            for w in wires:
                qml.RZ(theta[w], wires=w)
            for k, (i, j) in enumerate(pairs):
                qml.IsingZZ(theta[n_qubits + k], wires=[i, j])
            for w in wires:
                qml.Hadamard(wires=w)
        return qml.probs(wires=wires)

    return circuit


class QCBM(torch.nn.Module):
    def __init__(self, n_qubits: int, ansatz: str, layers: int):
        super().__init__()
        self.n_qubits, self.ansatz = n_qubits, ansatz
        shape = n_params(ansatz, n_qubits, layers)
        # iqp starts w/ smaller angles, big random ones made it start way off
        scale = 0.5 if ansatz == "iqp" else np.pi
        self.theta = torch.nn.Parameter(scale * (2 * torch.rand(shape, dtype=torch.float64) - 1))
        self.circuit = build_circuit(ansatz, n_qubits)

    def forward(self) -> torch.Tensor:
        return self.circuit(self.theta)          # prob of every bitstring, length 2^n

    @torch.no_grad()
    def sample(self, shots: int, generator=None) -> torch.Tensor:
        p = self.forward().clamp_min(0)          # clamp - tiny negative float errors
        idx = torch.multinomial(p / p.sum(), shots, replacement=True, generator=generator)
        weights = 2 ** torch.arange(self.n_qubits - 1, -1, -1)
        return ((idx.unsqueeze(1) // weights) % 2).double()     # index -> bits, (shots, n)


# ─────────────────────────────────────────────── Metrics ───────────────────────────────────────────────
# p = true dist, q = model (or counting)
def distribution_metrics(p: torch.Tensor, q: torch.Tensor, K: torch.Tensor, support: torch.Tensor) -> dict:
    eps = 1e-12   # avoid log(0)
    out = {
        "mmd": mmd_exact(p, q, K).item(),
        "tvd": 0.5 * (p - q).abs().sum().item(), # 0 same, 1 no overlap at all
        "kl": (p * (torch.log(p + eps) - torch.log(q + eps))).sum().item(),  # blows up if q=0 where p>0
        "fidelity": torch.sqrt(p * q).sum().pow(2).item(), # 1 = same
    }
    if len(support) < len(p): # validity only makes sense when some strings are "invalid" (bas)
        out["validity"] = q[support].sum().item() # how much prob is on valid strings
    return out


def coverage(samples: torch.Tensor, support: torch.Tensor) -> float:
    # what fraction of the valid strings show up at least once
    weights = 2 ** torch.arange(samples.shape[1] - 1, -1, -1)
    seen = set(((samples.long() * weights).sum(-1)).tolist())
    return len(seen & set(support.tolist())) / len(support)


# ─────────────────────────────────────────────── Training ──────────────────────────────────────────────
def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    samples_np, target_np, meta = dg.load(cfg.dataset, cfg.data_dir)
    n = meta["n_bits"]
    train_x = torch.tensor(samples_np, dtype=torch.float64)
    p_true = torch.tensor(target_np, dtype=torch.float64) # only used for grading, NOT training
    p_counts = empirical_distribution(train_x, n) # what frequency counting gives
    support = torch.nonzero(p_true > 1e-12).flatten()
    K = kernel_matrix(n, cfg.sigmas)

    model = QCBM(n, cfg.ansatz, cfg.layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # cosine lr - big steps at first, small at the end so it settles down
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr * 0.05)

    title = f"QCBM | dataset={cfg.dataset} ({n} qubits, {len(train_x)} training samples) | " \
            f"ansatz={cfg.ansatz} | {model.theta.numel()} parameters"
    print("=" * len(title) + f"\n{title}\n" + "=" * len(title))
    print(f"{'step':>6} {'train MMD':>11} {'TVD vs truth':>13} {'lr':>9}")

    best = {"loss": float("inf"), "theta": None, "step": 0}
    history, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        opt.zero_grad()
        loss = mmd_exact(model(), p_counts, K) # loss only uses the samples
        loss.backward()
        opt.step()
        sched.step()

        history.append(loss.item())
        # keep the best params (by train loss) in case it gets worse later
        if loss.item() < best["loss"] - 1e-9:
            best = {"loss": loss.item(), "theta": model.theta.detach().clone(), "step": step}
        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                tvd = 0.5 * (model() - p_true).abs().sum().item()   # just printing, doesnt affect training
            print(f"{step:>6} {loss.item():>11.6f} {tvd:>13.4f} {sched.get_last_lr()[0]:>9.4f}")
        if step - best["step"] >= cfg.patience:
            print(f"early stop at step {step} (no improvement for {cfg.patience} steps)")
            break
    train_time = time.time() - t0

    # ─────────────────────────────────────────────── Eval vs True distr ───────────────────────────────────────────────
    with torch.no_grad():
        model.theta.copy_(best["theta"])
        q_model = model()
        gen = torch.Generator().manual_seed(cfg.seed)
        shots = model.sample(cfg.eval_shots, gen)
        # fresh samples from the true dist (never trained on) for the sample-based test MMD
        test_x = torch.tensor(dg.index_to_bits(
            np.random.default_rng(cfg.seed + 1).choice(len(p_true), cfg.eval_shots, p=target_np), n),
            dtype=torch.float64)

    qcbm_m = distribution_metrics(p_true, q_model, K, support)
    base_m = distribution_metrics(p_true, p_counts, K, support)
    qcbm_m["coverage"] = coverage(shots, support)
    base_m["coverage"] = coverage(train_x, support)
    qcbm_m["test_mmd_samples"] = mmd_samples(shots, test_x, cfg.sigmas).item()

    result = {
        "config": asdict(cfg), "dataset_meta": meta, "n_parameters": model.theta.numel(),
        "steps_run": len(history), "best_step": best["step"], "train_seconds": round(train_time, 1),
        "final_train_mmd": best["loss"], "qcbm": qcbm_m, "counting_baseline": base_m,
        "loss_history_every_10": history[::10],
    }
    print_report(result)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.dataset}_{cfg.ansatz}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    torch.save(best["theta"], out / f"{stem}_params.pt") # so we can reload the trained circuit
    return result


def print_report(r: dict):
    q, b = r["qcbm"], r["counting_baseline"]
    rows = [("MMD (lower better)", "mmd"), ("TVD (lower better)", "tvd"), ("KL (lower better)", "kl"),
            ("Fidelity (higher better)", "fidelity"), ("Validity (higher better)", "validity"),
            ("Coverage (higher better)", "coverage")]
    print(f"\nResults vs TRUE distribution (best step {r['best_step']}, {r['train_seconds']}s)")
    print(f"{'metric':<26} {'QCBM':>10} {'counting':>10}")
    for label, key in rows:
        if key in q:
            print(f"{label:<26} {q[key]:>10.4f} {b[key]:>10.4f}")
    print(f"{'Test MMD from samples':<26} {q['test_mmd_samples']:>10.5f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # make a --flag for every field in Config automatically
    d = Config()
    for k, v in asdict(d).items():
        if k == "sigmas":
            ap.add_argument("--sigmas", type=float, nargs="+", default=list(v))
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = vars(ap.parse_args())
    args["sigmas"] = tuple(args["sigmas"])
    train(Config(**args))


if __name__ == "__main__":
    main()
