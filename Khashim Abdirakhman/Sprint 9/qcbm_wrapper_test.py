"""quick check of qcbm_wrapper.py (same idea as faisal's vae_wrapper_test.py)

python qcbm_wrapper_test.py
python qcbm_wrapper_test.py bitstrings.csv
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

from qcbm_wrapper import HERE, qcbm_function, to_bits
import data_generator as dg

"""
default csv = data/bas.csv (3x3 bas, 9 bits), or any csv w/ a "bitstring" column
checks the output format, hederik's list of lists input, the Team framework call, and that 128 bit data gets refused
"""

N_SHOW = 10
STEPS = 50 # just checking it runs, not a real training run
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "bas.csv"
if len(sys.argv) == 1 and not path.exists():
    subprocess.run([sys.executable, str(HERE / "data_generator.py"), "--all", "--out", str(HERE / "data")], check=True)


def check(output, n_rows, n_bits, label):
    assert output.dtype == torch.long, f"{label}: expected a torch.long tensor, like the other wrapped models"
    assert output.shape == (n_rows, n_bits), f"{label}: expected one generated bitstring per input row, same length"
    assert set(output.unique().tolist()) <= {0, 1}, f"{label}: only 0 and 1 are allowed"
    print(f"{label}: {tuple(output.shape)} tensor, dtype {output.dtype}, only 0/1 -> ok")


# CSV INPUT
x = to_bits(path)
n_rows, n_bits = x.shape
output = qcbm_function(path, steps=STEPS)
check(output, n_rows, n_bits, "csv input")

# LIST OF LISTS INPUT (hederik's format)
as_lists = x.long().tolist()
output_lists = qcbm_function(as_lists, steps=STEPS)
check(output_lists, n_rows, n_bits, "list of lists input")
#same data + same seed should give the same samples no matter the input format
assert torch.equal(output, output_lists), "csv and list of lists gave different results for the same data"

print(f"\nfirst {N_SHOW} generated bitstrings:")
for r in output[:N_SHOW]:
    print("".join(map(str, r.tolist())))

if n_bits == 9:
    valid = {tuple(p) for p in dg.bas_patterns(3).tolist()}
    pct = 100 * sum(tuple(r) in valid for r in output.tolist()) / len(output)
    print(f"\nvalid bars-and-stripes patterns: {pct:.1f}% (after only {STEPS} training steps)")

# TEAM FRAMEWORK CALL (evaluation_framework.py)
# the framework passes only these keywords, training_data is a torch.long tensor that's already split
t0 = time.time()
output_fw = qcbm_function(training_data=x.long(), num_epochs=50, seed=33, epsilon=0.3, num_samples=100)
check(output_fw, 100, n_bits, "framework-style call")
print(f"framework-style call took {time.time() - t0:.1f}s")

try:
    qcbm_function(num_epochs=50, seed=33)
    raise AssertionError("calling without data should have raised a ValueError")
except ValueError as e:
    print(f"no data -> ValueError: {e}")

# TOO MANY BITS
big = HERE.parents[1] / "Jacqui Person" / "Sprint8-bitstrings.csv"
if big.exists():
    t0 = time.time()
    try:
        qcbm_function(big)
        raise AssertionError("128-bit data should have raised a ValueError")
    except ValueError as e:
        print(f"\n128-bit csv refused in {time.time() - t0:.2f}s w/ ValueError: {e}")
else:
    print(f"\nskipped the 128-bit check, {big} not found")

#same check in the framework format (128-bit tensor), doesnt need jacqui's file
t0 = time.time()
try:
    qcbm_function(training_data=torch.zeros(10, 128, dtype=torch.long), num_epochs=50, seed=33, epsilon=0.3, num_samples=100)
    raise AssertionError("128-bit tensor should have raised a ValueError")
except ValueError as e:
    print(f"128-bit tensor refused in {time.time() - t0:.2f}s w/ ValueError: {e}")

print("\nall checks passed")
