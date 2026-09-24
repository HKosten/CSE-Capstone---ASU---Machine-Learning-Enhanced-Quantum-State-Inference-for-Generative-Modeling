import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformer_dice import calculate_mmd

#use pytest?
def test_identical():

    real_samples = torch.randint(0, 2, (200, 128))
    generated_samples = real_samples.clone()
    mmd = calculate_mmd(generated_samples, real_samples)

    assert abs(mmd) < 1e-6


