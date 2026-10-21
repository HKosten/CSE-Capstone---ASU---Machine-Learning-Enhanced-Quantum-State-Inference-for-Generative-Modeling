"""
vae_model.py  --  A VAE (variational autoencoder) for strings of 0s and 1s.

Contributor: Faisal Al-Qahtani
Project:     Machine Learning-Enhanced Quantum State Inference for Generative Modeling

How it works:
  * The ENCODER squeezes a string of bits into a few numbers.
  * The DECODER turns those numbers back into a string of bits.
  * To make a NEW string, we skip the encoder. We pick random numbers and let
    the decoder turn them into a string (it flips a coin for each bit).

How it learns (it tries to make its mistake score, the "loss", smaller):
  * Rebuild: the decoder's string should match the real string.
  * KL: the few numbers should look random, so random numbers work later.

The team's framework calls the function in vae_interface.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """VAE with a one-hidden-layer encoder and decoder (small on purpose: simple + fast)."""

    def __init__(self, n_bits: int = 128, hidden_dim: int = 256, latent_dim: int = 16):
        super().__init__()
        self.n_bits = n_bits
        self.latent_dim = latent_dim

        # Encoder: bitstring -> hidden -> (mu, logvar)
        self.enc_hidden = nn.Linear(n_bits, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> hidden -> one LOGIT per bit (sigmoid(logit) = P(bit = 1))
        self.dec_hidden = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, n_bits)

    # ---- the three building blocks -------------------------------------------------
    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_hidden(x))
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None):
        """z = mu + sigma * eps, eps ~ N(0, I). Written this way so gradients can flow
        through mu and sigma (the "reparameterization trick")."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits (NOT probabilities). Apply torch.sigmoid to get P(bit = 1)."""
        return self.dec_out(F.relu(self.dec_hidden(z)))

    # ---- full pass used during training --------------------------------------------
    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar

    # ---- generation ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate `n_samples` new bitstrings: z ~ N(0, I) -> decoder -> Bernoulli bits.
        Returns a float tensor of shape (n_samples, n_bits) containing only 0.0 / 1.0."""
        z = torch.randn((n_samples, self.latent_dim), generator=generator)
        probs = torch.sigmoid(self.decode(z))
        return torch.bernoulli(probs, generator=generator)


def vae_loss(logits, x, mu, logvar, beta: float = 1.0):
    """Negative ELBO, averaged over the batch.

    Returns (total, recon, kl); all three are "nats per bitstring".
    total = recon + beta * kl
    """
    # Sum over bits, then average over samples in the batch.
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    total = (recon + beta * kl).mean()
    return total, recon.mean(), kl.mean()


def train_vae(
    model: VAE,
    train_data: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
    generator: torch.Generator | None = None,
    verbose: bool = False,
) -> dict:
    """Train `model` in place. Returns per-epoch average losses:
    {"total": [...], "recon": [...], "kl": [...]}  (each list has `epochs` entries).

    `generator` controls the shuffling and the reparameterization noise, so passing a
    seeded torch.Generator makes training reproducible.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_data.shape[0]
    history = {"total": [], "recon": [], "kl": []}

    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)  # new shuffle every epoch
        sums = torch.zeros(3)
        for start in range(0, n, batch_size):
            x = train_data[order[start:start + batch_size]]
            optimizer.zero_grad()
            logits, mu, logvar = model(x, generator)
            total, recon, kl = vae_loss(logits, x, mu, logvar, beta)
            total.backward()
            optimizer.step()
            # weight by batch size so the last (smaller) batch is averaged correctly
            sums += torch.stack([total.detach(), recon.detach(), kl.detach()]) * x.shape[0]

        avg = (sums / n).tolist()
        history["total"].append(avg[0])
        history["recon"].append(avg[1])
        history["kl"].append(avg[2])
        if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
            print(f"  epoch {epoch:4d}/{epochs} | loss {avg[0]:8.3f} | recon {avg[1]:8.3f} | kl {avg[2]:7.3f}")

    return history


@torch.no_grad()
def evaluate_loss(model: VAE, data: torch.Tensor, beta: float = 1.0,
                  generator: torch.Generator | None = None) -> float:
    """Average negative ELBO on `data` (e.g. the held-out set). Lower = better fit."""
    model.eval()
    logits, mu, logvar = model(data, generator)
    total, _, _ = vae_loss(logits, data, mu, logvar, beta)
    return total.item()
