"""
vae_wrapper_test.py  --  quick check of the wrapper, the same idea as the transformer's wrapper test.

    python vae_wrapper_test.py                    # uses bars_stripes_128.csv in this folder
    python vae_wrapper_test.py bitstrings.csv     # or any CSV with a "bitstring" column

It runs vae_function on the CSV, checks the output format, and prints the first generated
bitstrings as plain strings (change N_SHOW to print more).
"""

import sys

import torch

from vae_interface import load_bitstrings
from vae_wrapper import vae_function

N_SHOW = 10
path = sys.argv[1] if len(sys.argv) > 1 else "bars_stripes_128.csv"

n_rows, n_bits = load_bitstrings(path).shape
output = vae_function(path)

assert output.dtype == torch.long, "expected a torch.long tensor, like the other wrapped models"
assert output.shape == (n_rows, n_bits), "expected one generated bitstring per input row, same length"
assert set(output.unique().tolist()) <= {0, 1}, "only 0 and 1 are allowed"

print(f"Output: {tuple(output.shape)} tensor, dtype {output.dtype}, only 0/1\n")
for r in output[:N_SHOW]:
    print("".join(map(str, r.tolist())))
