from sklearn.model_selection import train_test_split
from ignite.metrics import MaximumMeanDiscrepancy
import torch
import random
import pandas as pd
import numpy as np


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

    data = torch.tensor(pd.read_csv(settings["training_data"], header=None).values,dtype=torch.long)

    train_data, test_data = train_test_split(data, train_size=settings["epsilon"], random_state=settings["seed"])

    model_settings = {**settings, "training_data": train_data}
    generated_samples = model_function(**model_settings)

    count = min(len(generated_samples), len(test_data))
    mmd = calculate_mmd(generated_samples[:count], test_data[:count])

    torch.set_printoptions(threshold=float("inf"))

    print("Ouput:\n")

    #works for both list and tensor
    for r in generated_samples:
        if isinstance(r, torch.Tensor):
            r = r.tolist()
        print("".join(map(str, r)))
    
    print("MMD:", mmd)

    return mmd
