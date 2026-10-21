"""
vae_wrapper.py  --  Faisal's VAE in the Team framework format.

Main.py runs it the same way as the other models:

    import vae_wrapper as vw
    ef.evaluate_model(vw.vae_function, settings)

evaluation_framework.py reads the CSV, keeps `epsilon` of the rows for training, and then calls
vae_function(**settings) with:
    training_data   torch.long tensor (n_train, n_bits) of 0/1 -- already the training split
    num_epochs      number of passes over the training rows
    seed            random seed (same seed -> same output)
    epsilon         training fraction; the framework has already applied it, so it is not used here
    num_samples     how many bitstrings to generate

It returns a torch.long tensor (num_samples, n_bits) of 0/1, the same format as the transformer wrapper.
vae_model.py and vae_interface.py are unchanged copies of the files in "Faisal Al-Qahtani/CSE486/VAE".
"""

import torch

from vae_interface import run_vae


def vae_function(**settings) -> torch.Tensor:
    out = run_vae(
        settings["training_data"],
        seed=settings["seed"],
        train_frac=1.0,  # the framework already split the data using epsilon
        epochs=settings["num_epochs"],
        n_samples=settings["num_samples"],
        verbose=True,  # prints the loss 10 times during training, like the other models
    )
    return out["generated_samples"].long()
