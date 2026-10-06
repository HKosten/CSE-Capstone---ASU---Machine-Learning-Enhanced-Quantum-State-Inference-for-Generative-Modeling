import torch
from transformer_wrapper import transformer_function

output = transformer_function("bitstrings.csv")

torch.set_printoptions(threshold=float("inf"))

print("Ouput:\n")

for r in output:
    print("".join(map(str, r.tolist())))