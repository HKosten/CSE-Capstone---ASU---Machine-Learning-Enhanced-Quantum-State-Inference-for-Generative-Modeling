from sklearn.model_selection import train_test_split
from ignite.metrics import MaximumMeanDiscrepancy
import torch
import random
import pandas as pd
import numpy as np


def load_data(training_data):
    
    df = pd.read_csv(training_data, dtype={"bitstring": str})
    return torch.tensor([[int(bit) for bit in s] for s in df["bitstring"]], dtype=torch.long)


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_mmd(generated_samples, real_samples):
    
    mmd = MaximumMeanDiscrepancy()
    mmd.update((torch.as_tensor(generated_samples, dtype=torch.float32), torch.as_tensor(real_samples, dtype=torch.float32)))
    return mmd.compute()


def evaluate_model(model_function, settings):
    
    set_seed(settings["seed"])

    data = load_data(settings["training_data"])

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=settings["seed"])

    num_train = max(1, int(len(train_data) * settings["epsilon"]))
    train_data = train_data[:num_train]

    model_settings = {**settings, "training_data": train_data}
    generated_samples = model_function(**model_settings)

    count = min(len(generated_samples), len(test_data))
    mmd = calculate_mmd(generated_samples[:count], test_data[:count])

    torch.set_printoptions(threshold=float("inf"))

    print("Ouput:\n")

    for r in generated_samples:
        print("".join(map(str, r.tolist())))
    
    print("MMD:", mmd)

    return mmd
