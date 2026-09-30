"""MMD for bitstrings (pytorch). run this file to do a quick sanity check"""

import itertools
from typing import Sequence

import torch

# MMD = how different 2 distributions are. 0 -> same, bigger -> more different
# kernel is gaussian: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
# for 0/1 strings ||x-y||^2 is just the # of bits that differ (hamming dist.)
# same kernel form as De Luca's iqpopt so we can compare numbers w/ it
# using a few sigmas and averaging them -> small sigma cares about exact matches, big sigma about overall shape
DEFAULT_SIGMAS = (0.5, 1.0, 2.0)


def all_bitstrings(n_bits: int, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0, 1], repeat=n_bits)), dtype=dtype)


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor,
                    sigmas: Sequence[float] = DEFAULT_SIGMAS) -> torch.Tensor:
    # kernel matrix between every row of X and every row of Y, avg over sigmas
    X, Y = X.double(), Y.double()
    d2 = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1) # (len X, len Y) hamming dists
    return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas) / len(sigmas)


def kernel_matrix(n_bits: int, sigmas: Sequence[float] = DEFAULT_SIGMAS) -> torch.Tensor:
    # kernel for all 2^n strings vs all 2^n strings, only ok for small n (used by mmd_exact)
    B = all_bitstrings(n_bits)
    return gaussian_kernel(B, B, sigmas)


def mmd_exact(p: torch.Tensor, q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # when we have the full prob. vectors: MMD^2 = (p-q)^T K (p-q)
    diff = p.double() - q.double()
    return diff @ K @ diff


def mmd_samples(X: torch.Tensor, Y: torch.Tensor, sigmas: Sequence[float] = DEFAULT_SIGMAS, unbiased: bool = True) -> torch.Tensor:
    # same thing but from samples only (rows = bitstrings), works for any n
    # unbiased -> skip each sample compared w/ itself (the diagonal)
    # so 2 sample sets from the same dist. give ~0 (can go a tiny bit negative, thats fine)
    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = gaussian_kernel(X, X, sigmas), gaussian_kernel(Y, Y, sigmas), gaussian_kernel(X, Y, sigmas)
    if unbiased:
        xx = (Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
        yy = (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
    else:
        xx, yy = Kxx.mean(), Kyy.mean()
    return xx + yy - 2 * Kxy.mean()


def median_heuristic(X: torch.Tensor) -> float:
    # usual default for sigma = median dist. between pairs of samples
    X = X.double()
    d = torch.cdist(X, X)
    iu = torch.triu_indices(len(X), len(X), offset=1)   # upper triangle only, no dupes/self pairs
    return d[iu[0], iu[1]].median().item()


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
