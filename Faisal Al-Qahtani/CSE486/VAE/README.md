# Variational Autoencoder (VAE)

**Contributor:** Faisal Al-Qahtani  
**Project:** Machine Learning-Enhanced Quantum State Inference for Generative Modeling

## Overview

This part of the project focuses on using a Variational Autoencoder (VAE) to learn patterns in binary datasets and generate new bitstrings.

Following the sponsor’s feedback, I am moving away from the Bayesian/Beta-Bernoulli approach. The goal is to develop a working generator that can be compared with the team’s other models using the same datasets and evaluation settings.

## Why VAE?

The sponsor suggested considering a VAE or Wasserstein GAN (WGAN). My comparison focuses on implementation complexity, training behavior, and support for binary outputs.

| Consideration | VAE | WGAN |
| --- | --- | --- |
| Main components | Encoder and decoder | Generator and critic |
| Training approach | Reconstruction loss and latent-space regularization | Alternating generator and critic updates |
| Binary outputs | Decoder probabilities can define Bernoulli outputs | Discrete sampling requires care during training |
| Implementation | A practical starting point for this project | Requires additional attention to adversarial training and critic constraints |

VAE is my chosen direction because it provides a straightforward way to model binary data and fits the time available for implementation. This is an implementation decision, not a claim that it will outperform the other models.

## Planned Model Design

1. **Encoder:** Takes a binary input and produces the parameters of a latent probability distribution.
2. **Latent sampling:** Samples a latent representation using the reparameterization trick during training.
3. **Decoder:** Converts the latent representation into a probability for each output bit.
4. **Binary generation:** Samples each output bit from its predicted Bernoulli probability.

The planned training objective combines binary reconstruction loss with KL divergence, which regularizes the latent distribution.

## Dataset

The initial dataset is the Bars-and-Stripes dataset prepared for the project:

- Grid size: 8 × 16.
- Bitstring length: 128.
- Number of unique samples: 1,000.
- Dataset-generation seed: 50.
- Patterns: Horizontal stripes or vertical bars.

The model will train on a selected portion of the data, with held-out samples reserved for evaluation.

## Implementation Goals

- Implement the VAE in PyTorch.
- Train it on a shared binary dataset.
- Generate 1,000 bitstrings matching the input length.
- Evaluate the generated samples using the team’s shared MMD implementation.
- Commit the implementation and document the experiment settings.

## Evaluation Plan

Compare the generated samples with held-out data using maximum mean discrepancy (MMD).

Also evaluate a baseline of 1,000 random bitstrings using the same reference data and MMD settings. The target is for the VAE to achieve a lower MMD than the random baseline.

Record the training loss, MMD results, random seed, training split, and epoch count. Results are pending.

## Team Integration

Provide a function that accepts the team’s agreed inputs and returns generated bitstrings and training-loss information.

Check that repeated runs with the same seed produce identical outputs under the same execution settings. Keep the interface in a separate file so it can be updated easily if the team’s requirements change.

## Current Status

VAE is the selected direction. Implementation, training results, and integration checks still need to be documented.
