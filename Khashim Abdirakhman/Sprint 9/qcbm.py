"""QCBM trained w/ MMD (sprint 8)

python data_generator.py --all --out data
python qcbm.py --dataset bas --ansatz strong --layers 10 --steps 400
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

import data_generator as dg
from mmd import DEFAULT_SIGMAS, empirical_distribution, kernel_matrix, mmd_exact, mmd_samples
"""
QCBM = quantum circuit used as a generative model measure the circuit -> get bitstrings, prob of each one = |amplitude|^2 (born rule)
We tune the angles so the circuit's dist. matches the data
changes from sprint 7 (Oct 7):
    - FIX coverage: qcbm got 2000 samples but counting only had the 1000 (or 100) training ones, now everyone gets the same #
    - new baseline "smoothed" = counting + 0.5 fake count on every bitstring, shows if the qcbm wins just bc it doesnt put 0 anywhere
    - test MMD from samples for counting + smoothed too data seed and init seed are separate now
    - train() can take samples directly -> stability.py doesnt need a csv per run
    - output names have n + seed in them so runs dont overwrite each other
    - validity only for bas
    - training loop moved into fit() (only needs samples), train() calls it. qcbm_wrapper.py uses it too

"""
"""
QCBM = quantum circuit used as a generative model measure the circuit -> get bitstrings, prob of each one = |amplitude|^2 (born rule)
We tune the angles so the circuit's dist. matches the data

changes from sprint 7 (Oct 7):
- FIX coverage: qcbm got 2000 samples but counting only had the 1000 (or 100) training ones, now everyone gets the same #
- new baseline "smoothed" = counting + 0.5 fake count on every bitstring, shows if the qcbm wins just bc it doesnt put 0 anywhere
- test MMD from samples for counting + smoothed too
-data seed and init seed are separate now
- train() can take samples directly -> stability.py doesnt need a csv per run
- output names have n + seed in them so runs dont overwrite each other
- validity only for bas
- training loop moved into fit() (only needs samples), train() calls it. qcbm_wrapper.py uses it too
"""


# CONFIG
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
    test_shots: int = 2000 # fresh samples from the true dist. for the test MMD
    smooth_alpha: float = 0.5
    seed: int = 42
    log_every: int = 50
    out_dir: str = "results"
    tag: str = ""


def run_name(cfg: Config, n_train: int) -> str:
    return f"{cfg.dataset}_{cfg.ansatz}_n{n_train}_s{cfg.seed}{cfg.tag}"


# CIRCUITS
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
            """
            iqp: H on everything -> Z and ZZ phases -> H again
            (H exp(iθZ) H = exp(iθX) so this is the same as the exp(iθX..) gates in iqpopt)
            iqpopt uses MultiRZ(2θ) -> iqpopt params = these / 2 (iqpopt_compare.py checks it)
            """
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


# SAMPLING HELPERS
def index_to_bits_t(idx: torch.Tensor, n_bits: int) -> torch.Tensor:
    weights = 2 ** torch.arange(n_bits - 1, -1, -1)
    return ((idx.unsqueeze(1) // weights) % 2).double()     # index -> bits, (shots, n)


def sample_bits(p: torch.Tensor, shots: int, generator=None) -> torch.Tensor:
    # draw bitstrings from any prob. vector (qcbm, counting, smoothed)
    p = p.detach().double().clamp_min(0) # clamp - tiny negative float errors
    idx = torch.multinomial(p / p.sum(), shots, replacement=True, generator=generator)
    return index_to_bits_t(idx, len(p).bit_length() - 1)


def smoothed(p_counts: torch.Tensor, n_train: int, alpha: float) -> torch.Tensor:
    # (count + alpha) / (N + alpha * 2^n), alpha=0.5 is the standard add-1/2 (KT) estimator
    return (p_counts * n_train + alpha) / (n_train + alpha * len(p_counts))


# METRICS
# p = true dist, q = model (or counting)
def distribution_metrics(p: torch.Tensor, q: torch.Tensor, K: torch.Tensor, support=None) -> dict:
    eps = 1e-12   # avoid log(0)
    out = {
        "mmd": mmd_exact(p, q, K).item(),
        "tvd": 0.5 * (p - q).abs().sum().item(), # 0 same, 1 no overlap at all
        "kl": (p * (torch.log(p + eps) - torch.log(q + eps))).sum().item(),  # blows up if q=0 where p>0
        "fidelity": torch.sqrt(p * q).sum().pow(2).item(), # 1 = same
    }
    if support is not None: # validity only makes sense when some strings are "invalid" (bas)
        out["validity"] = q[support].sum().item() # how much prob is on valid strings
    return out


def coverage(samples: torch.Tensor, support: torch.Tensor) -> float:
    # what fraction of the valid strings show up at least once
    weights = 2 ** torch.arange(samples.shape[1] - 1, -1, -1)
    seen = set(((samples.long() * weights).sum(-1)).tolist())
    return len(seen & set(support.tolist())) / len(support)


# TRAINING
def fit(train_x: torch.Tensor, cfg: Config, p_true: torch.Tensor = None, verbose: bool = True) -> QCBM:
    # training loop, only needs the samples. p_true is just for the TVD column in the log
    log = print if verbose else (lambda *a, **k: None)
    train_x = train_x.double()
    n, n_train = train_x.shape[1], len(train_x)
    p_counts = empirical_distribution(train_x, n)
    K = kernel_matrix(n, cfg.sigmas)

    torch.manual_seed(cfg.seed)
    model = QCBM(n, cfg.ansatz, cfg.layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # cosine lr - big steps at first, small at the end so it settles down
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr * 0.05)

    title = f"QCBM | {n} qubits, {n_train} training samples | " \
            f"ansatz={cfg.ansatz} | {model.theta.numel()} parameters | seed={cfg.seed}"
    log("=" * len(title) + f"\n{title}\n" + "=" * len(title))
    log(f"{'step':>6} {'train MMD':>11}" + (f" {'TVD vs truth':>13}" if p_true is not None else "") + f" {'lr':>9}")

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
        if verbose and (step % cfg.log_every == 0 or step == 1):
            line = f"{step:>6} {loss.item():>11.6f}"
            if p_true is not None:
                with torch.no_grad():
                    tvd = 0.5 * (model() - p_true).abs().sum().item()   # just printing, doesnt affect training
                line += f" {tvd:>13.4f}"
            log(line + f" {sched.get_last_lr()[0]:>9.4f}")
        if step - best["step"] >= cfg.patience:
            log(f"early stop at step {step} (no improvement for {cfg.patience} steps)")
            break

    with torch.no_grad():
        model.theta.copy_(best["theta"])
    #train() needs these for the json
    model.history, model.best = history, best
    model.train_seconds = time.time() - t0
    return model


def train(cfg: Config, train_x: torch.Tensor = None, p_true: torch.Tensor = None, meta: dict = None,
          save: bool = True, verbose: bool = True) -> dict:
    # either loads <data_dir>/<dataset>.csv OR uses train_x / p_true / meta passed in (stability.py does that)
    if train_x is None:
        samples_np, target_np, meta = dg.load(cfg.dataset, cfg.data_dir)
        train_x = torch.tensor(samples_np, dtype=torch.float64)
        p_true = torch.tensor(target_np, dtype=torch.float64) # only used for grading, NOT training
    train_x, p_true = train_x.double(), p_true.double()
    n, n_train = meta["n_bits"], len(train_x)
    is_bas = "valid_patterns" in meta
    p_counts = empirical_distribution(train_x, n) # what frequency counting gives
    p_smooth = smoothed(p_counts, n_train, cfg.smooth_alpha)
    support = torch.nonzero(p_true > 1e-12).flatten()
    K = kernel_matrix(n, cfg.sigmas)

    if verbose:
        print(f"dataset={cfg.dataset}")
    model = fit(train_x, cfg, p_true, verbose)
    history, best, train_time = model.history, model.best, model.train_seconds

    # EVAL VS TRUE DISTR
    with torch.no_grad():
        q_model = model()
    gen = torch.Generator().manual_seed(cfg.seed)
    """
    fresh samples from the true dist (never trained on) for the sample-based test MMD
    10_000 + seed so its never the same rng stream as the training samples
    """
    test_rng = np.random.default_rng(10_000 + cfg.seed)
    test_x = torch.tensor(dg.index_to_bits(
        test_rng.choice(len(p_true), cfg.test_shots, p=p_true.numpy() / p_true.numpy().sum()), n),
        dtype=torch.float64)

    metrics = {}
    for name, q in [("qcbm", q_model), ("counting", p_counts), ("smoothed", p_smooth)]:
        m = distribution_metrics(p_true, q, K, support if is_bas else None)
        #same # of draws as training samples for everyone. counting = the training samples themselves
        draws = train_x if name == "counting" else sample_bits(q, n_train, gen)
        m["coverage"] = coverage(draws, support)
        m["test_mmd_samples"] = mmd_samples(sample_bits(q, cfg.test_shots, gen), test_x, cfg.sigmas).item()
        metrics[name] = m

    result = {
        "config": asdict(cfg), "dataset_meta": meta, "n_train": n_train,
        "n_parameters": model.theta.numel(), "steps_run": len(history), "best_step": best["step"],
        "train_seconds": round(train_time, 1), "final_train_mmd": best["loss"],
        **metrics, "loss_history_every_10": history[::10],
    }
    if verbose:
        print_report(result)

    if save:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = run_name(cfg, n_train)
        (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
        torch.save(best["theta"], out / f"{stem}_params.pt") # so we can reload the trained circuit
    return result


def print_report(r: dict):
    rows = [("MMD (lower better)", "mmd"), ("TVD (lower better)", "tvd"), ("KL (lower better)", "kl"),
            ("Fidelity (higher better)", "fidelity"), ("Validity (higher better)", "validity"),
            ("Coverage (higher better)", "coverage"), ("Test MMD from samples", "test_mmd_samples")]
    print(f"\nResults vs TRUE distribution (best step {r['best_step']}, {r['train_seconds']}s)")
    print(f"{'metric':<26} {'QCBM':>10} {'counting':>10} {'smoothed':>10}")
    for label, key in rows:
        if key in r["qcbm"]:
            print(f"{label:<26} " + " ".join(f"{r[m][key]:>10.4f}" for m in ("qcbm", "counting", "smoothed")))
    print()


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
