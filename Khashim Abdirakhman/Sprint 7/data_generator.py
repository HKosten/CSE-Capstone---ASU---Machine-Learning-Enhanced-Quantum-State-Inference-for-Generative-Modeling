"""DB generator for QCBM

python data_generator.py --dataset bas --size 3 --n-samples 1000 --out data
python data_generator.py --all --out data
"""

import argparse
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Every dataset here we know the EXACT probs for, so we can check the model against the real dist. and not just the samples it trained on
#
# Bas - bars and stripes on n x n grid
# Markov - each bit depends on the one before it, P(n | n-1) (same as README problem)
# Gaussian - 2 bell curves over 0..2^n-1, written as n bit binary
# Quantum - measurements from a fixed random circuit in pennylane (closest to real quantum data)
#
# Save:  <name>.csv (col "bitstring", same format as faisal's + jacqui's generators)
#        <name>_target.npy (exact prob of every bitstring)
#        <name>_meta.json (settings, so we can make the same data again)


@dataclass
class Dataset:
    name: str
    n_bits: int
    target: np.ndarray # exact probs, length 2^n_bits
    settings: dict = field(default_factory=dict)

    @property
    def support(self) -> np.ndarray:
        # indices of bitstrings w/ prob > 0
        return np.flatnonzero(self.target > 1e-12)

    def sample(self, n_samples: int, seed: int = 0) -> np.ndarray:
        # draw from the exact dist -> (n_samples, n_bits) array of 0s and 1s
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.target), size=n_samples, p=self.target)
        return index_to_bits(idx, self.n_bits)


# ─────────────────────────────────────────────── Bitstring helpers ─────────────────────────────────────
# bit 0 = most significant bit, same order pennylane uses for qml.probs
def index_to_bits(idx, n_bits: int) -> np.ndarray:
    idx = np.atleast_1d(idx)
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((idx[:, None] >> shifts) & 1).astype(np.int8)


def bits_to_index(bits: np.ndarray) -> np.ndarray:
    bits = np.atleast_2d(bits)
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1)
    return bits.astype(np.int64) @ weights


def all_bitstrings(n_bits: int) -> np.ndarray:
    return index_to_bits(np.arange(2 ** n_bits), n_bits)


# ─────────────────────────────────────────────── Bars and stripes ───────────────────────────────────
def bars_and_stripes(size: int = 3) -> Dataset:
    n_bits = size * size
    patterns = set()
    for bits in itertools.product([0, 1], repeat=size):
        rows = np.repeat(np.array(bits)[:, None], size, axis=1) # horizontal stripes
        patterns.add(tuple(rows.flatten()))
        patterns.add(tuple(rows.T.flatten())) # transpose -> vertical bars
    # set removes dupes (all 0s / all 1s show up as both a bar and a stripe)
    idx = bits_to_index(np.array(sorted(patterns)))
    target = np.zeros(2 ** n_bits)
    target[idx] = 1.0 / len(idx) # uniform over valid ones, 0 everywhere else
    return Dataset("bas", n_bits, target, {"size": size, "valid_patterns": len(idx)})


# ─────────────────────────────────────────────── Markov chain, P(bit n | bit n-1) ───────────────────
def markov_chain(n_bits: int = 8, p_first_1: float = 0.5,
                 p_1_given_0: float = 0.3, p_1_given_1: float = 0.8) -> Dataset:
    bits = all_bitstrings(n_bits)
    # prob of each string = P(first bit) * P(bit1|bit0) * P(bit2|bit1) * ...
    p = np.where(bits[:, 0] == 1, p_first_1, 1 - p_first_1).astype(float)
    for i in range(1, n_bits):
        p_one = np.where(bits[:, i - 1] == 1, p_1_given_1, p_1_given_0)
        p *= np.where(bits[:, i] == 1, p_one, 1 - p_one)
    return Dataset("markov", n_bits, p, {
        "p_first_1": p_first_1, "p_1_given_0": p_1_given_0, "p_1_given_1": p_1_given_1})


# ─────────────────────────────────────────────── Gaussian mixture ───────────────────────────────────
def gaussian_mixture(n_bits: int = 8, centers=(0.3, 0.7), width: float = 0.06,
                     weights=(0.5, 0.5)) -> Dataset:
    x = np.arange(2 ** n_bits) / 2 ** n_bits # 0..1
    p = sum(w * np.exp(-0.5 * ((x - c) / width) ** 2) for c, w in zip(centers, weights))
    p /= p.sum()
    return Dataset("gaussian", n_bits, p, {
        "centers": list(centers), "width": width, "weights": list(weights)})


# ─────────────────────────────────────────────── Random quantum circuit ─────────────────────────────
def quantum_circuit(n_bits: int = 8, n_layers: int = 2, seed: int = 7) -> Dataset:
    import pennylane as qml

    # fixed seed so the "true" circuit is always the same one
    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_bits)
    weights = rng.uniform(0, 2 * np.pi, size=shape)
    dev = qml.device("default.qubit", wires=n_bits)

    @qml.qnode(dev)
    def circuit():
        qml.StronglyEntanglingLayers(weights, wires=range(n_bits))
        return qml.probs(wires=range(n_bits))

    p = np.asarray(circuit(), dtype=float)
    return Dataset("quantum", n_bits, p / p.sum(), {"n_layers": n_layers, "seed": seed})


BUILDERS = {
    "bas": lambda a: bars_and_stripes(a.size),
    "markov": lambda a: markov_chain(a.n_bits),
    "gaussian": lambda a: gaussian_mixture(a.n_bits),
    "quantum": lambda a: quantum_circuit(a.n_bits),
}


# ─────────────────────────────────────────────── Save & Load ───────────────────────────────────────────
def save(ds: Dataset, samples: np.ndarray, out_dir, seed: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{ds.name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring"])
        for row in samples:
            w.writerow(["".join(map(str, row))])
    np.save(out / f"{ds.name}_target.npy", ds.target)
    meta = {"name": ds.name, "n_bits": ds.n_bits, "n_samples": len(samples),
            "seed": seed, "support_size": int(len(ds.support)), **ds.settings}
    (out / f"{ds.name}_meta.json").write_text(json.dumps(meta, indent=2))
    return out / f"{ds.name}.csv"


def load(name: str, data_dir):
    # returns (samples, target, meta)
    d = Path(data_dir)
    with open(d / f"{name}.csv") as f:
        rows = [r["bitstring"] for r in csv.DictReader(f)]
    samples = np.array([[int(c) for c in s] for s in rows], dtype=np.int8)
    target = np.load(d / f"{name}_target.npy")
    meta = json.loads((d / f"{name}_meta.json").read_text())
    return samples, target, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=BUILDERS, default="bas")
    ap.add_argument("--all", action="store_true", help="generate every dataset")
    ap.add_argument("--size", type=int, default=3, help="grid size for bas")
    ap.add_argument("--n-bits", type=int, default=8, help="bits for markov/gaussian/quantum")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    for name in (BUILDERS if args.all else [args.dataset]):
        ds = BUILDERS[name](args)
        path = save(ds, ds.sample(args.n_samples, args.seed), args.out, args.seed)
        print(f"{name:<9} {ds.n_bits} bits | {len(ds.support):>4} of {2 ** ds.n_bits} "
              f"bitstrings possible | {args.n_samples} samples -> {path}")


if __name__ == "__main__":
    main()
