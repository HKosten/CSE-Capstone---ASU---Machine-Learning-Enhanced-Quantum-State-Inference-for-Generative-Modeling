"""MMD for bitstrings (pytorch). run this file to do a quick sanity check

python mmd.py
"""

import itertools
import math
from typing import Sequence

import torch
"""
MMD = how different 2 distributions are. 0 -> same, bigger -> more different kernel is gaussian: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
for 0/1 strings ||x-y||^2 is just the # of bits that differ (hamming dist.)

Same kernel form as Sponsor's iqpopt so we can compare numbers w/ it using a few sigmas and averaging them -> small sigma cares about exact matches,
big sigma about overall shape changes from sprint 7:
    - kernel uses |x|^2 + |y|^2 - 2x.y, same numbers but ~10x less memory (2000 x 2000 test samples was ~300MB per call)
    - mmd_samples needs at least 2 samples (unbiased version divides by m-1)

changes from sprint 9 (Nov 4):
    - mmd_exact_product: the gaussian kernel on 0/1 strings is a product over bits, exp(-d/(2s^2)) = prod_i exp(-|x_i-y_i|/(2s^2)),
      so K @ v is n small 2x2 steps. same numbers as mmd_exact, but no 2^n x 2^n matrix (16 bits was 32 GB)
    - median_sigma / iqpopt_sigmas: kernel widths from the data instead of the fixed (0.5, 1, 2), same recipe as sponsor's iqpopt

"""
DEFAULT_SIGMAS = (0.5, 1.0, 2.0)


def all_bitstrings(n_bits: int, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0, 1], repeat=n_bits)), dtype=dtype)


def sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # (len X, len Y) hamming dists w/o building a (len X, len Y, n_bits) tensor
    X, Y = X.double(), Y.double()
    d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T
    return d2.clamp_min(0)


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor,
                    sigmas: Sequence[float] = DEFAULT_SIGMAS) -> torch.Tensor:
    # kernel matrix between every row of X and every row of Y, avg over sigmas
    d2 = sq_dists(X, Y)
    return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)


def kernel_matrix(n_bits: int, sigmas: Sequence[float] = DEFAULT_SIGMAS) -> torch.Tensor:
    # kernel for all 2^n strings vs all 2^n strings, only ok for small n (used by mmd_exact)
    B = all_bitstrings(n_bits)
    return gaussian_kernel(B, B, sigmas)


def mmd_exact(p: torch.Tensor, q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # when we have the full prob. vectors: MMD^2 = (p-q)^T K (p-q)
    diff = p.double() - q.double()
    return diff @ K @ diff


def kernel_apply(v: torch.Tensor, n_bits: int, sigma: float) -> torch.Tensor:
    # K_sigma @ v without building K. K = kron of n copies of [[1, a], [a, 1]], a = exp(-1/(2 sigma^2))
    a = math.exp(-1.0 / (2 * sigma ** 2))
    for i in range(n_bits):
        v = v.reshape(2 ** i, 2, 2 ** (n_bits - i - 1))
        v = v + a * v.flip(1)   # 2x2 step on bit i
    return v.reshape(-1)


def mmd_exact_product(p: torch.Tensor, q: torch.Tensor, n_bits: int, sigmas: Sequence[float] = DEFAULT_SIGMAS) -> torch.Tensor:
    # same as mmd_exact (avg over sigmas) but memory is 2^n, not 4^n. used for the loss at n > 12
    diff = p.double() - q.double()
    return sum(diff @ kernel_apply(diff, n_bits, s) for s in sigmas) / len(sigmas)


def mmd_samples(X: torch.Tensor, Y: torch.Tensor, sigmas: Sequence[float] = DEFAULT_SIGMAS, unbiased: bool = True) -> torch.Tensor:
    """
    same thing but from samples only (rows = bitstrings), works for any n
    unbiased -> skip each sample compared w/ itself (the diagonal)
    so 2 sample sets from the same dist. give ~0 (can go a tiny bit negative, thats fine)
    """
    m, n = len(X), len(Y)
    if unbiased and (m < 2 or n < 2):
        raise ValueError("unbiased MMD needs at least 2 samples in each set")
    Kxx, Kyy, Kxy = gaussian_kernel(X, X, sigmas), gaussian_kernel(Y, Y, sigmas), gaussian_kernel(X, Y, sigmas)
    if unbiased:
        xx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
        yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    else:
        xx, yy = Kxx.mean(), Kyy.mean()
    return xx + yy - 2 * Kxy.mean()


def median_heuristic(X: torch.Tensor) -> float:
    # usual default for sigma = median dist. between pairs of samples
    d = sq_dists(X, X).sqrt()
    iu = torch.triu_indices(len(X), len(X), offset=1)   # upper triangle only, no dupes/self pairs
    return d[iu[0], iu[1]].median().item()


def median_sigma(X: torch.Tensor, n_max: int = 1000) -> float:
    # sponsor's iqpopt: sqrt of the median L2 distance. for 0/1 strings L2 = sqrt(hamming), so this is hamming^(1/4)
    return median_heuristic(X[:n_max].double()) ** 0.5


def iqpopt_sigmas(X: torch.Tensor, n_sigmas: int = 1) -> list:
    # iqpopt sigma_heuristic: from "mean operator weight 2" up to the median heuristic
    n = X.shape[-1]
    med = median_sigma(X)
    low = math.sqrt(-1 / (2 * math.log(1 - 4 / n))) if n > 4 else med
    return torch.linspace(low, med, n_sigmas).tolist()


def empirical_distribution(samples: torch.Tensor, n_bits: int) -> torch.Tensor:
    # samples -> prob. vector of length 2^n (this is basically frequency counting)
    weights = 2 ** torch.arange(n_bits - 1, -1, -1)
    idx = (samples.long() * weights).sum(-1)            # bits -> int index
    counts = torch.bincount(idx, minlength=2 ** n_bits).double()
    return counts / counts.sum()


if __name__ == "__main__":
    # sanity check - same dist. should be ~0, diff. dist. should be clearly > 0
    torch.manual_seed(0)
    n = 8
    a1 = (torch.rand(500, n) < 0.2).double()
    a2 = (torch.rand(500, n) < 0.2).double()
    b = (torch.rand(500, n) < 0.6).double()
    print(f"median heuristic sigma      : {median_heuristic(a1):.3f}")
    print(f"MMD same distribution       : {mmd_samples(a1, a2).item():+.5f}")
    print(f"MMD different distribution  : {mmd_samples(a1, b).item():+.5f}")
    K = kernel_matrix(n)
    pa, pb = empirical_distribution(a1, n), empirical_distribution(b, n)
    # these 2 should match exactly (biased sample version == exact version on the counts)
    print(f"exact vs biased sample MMD  : {mmd_exact(pa, pb, K).item():.5f} vs "
          f"{mmd_samples(a1, b, unbiased=False).item():.5f} (should match)")
    # product kernel has to give the same number as the matrix
    print(f"matrix vs product kernel    : {mmd_exact(pa, pb, K).item():.12f} vs "
          f"{mmd_exact_product(pa, pb, n).item():.12f} (should match)")
