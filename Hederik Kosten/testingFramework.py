import random

import LSTM
import pytorchMMD
import torch

with open("Hederik Kosten\\chr1.fa", 'r') as f:
    lines = f.readlines()

def encode(text):
    encoding = {
        'A': '00',
        'C': '01',
        'G': '10',
        'T': '11'
    }

    return [
        int(bit)
        for base in text.upper()
        if base in encoding
        for bit in encoding[base]
    ]

dna_sequences = [
    line.strip().upper()
    for line in lines
    if line.strip()
]

# Remove any sequence containing N
dna_sequences = [
    sequence
    for sequence in dna_sequences
    if 'N' not in sequence
]

# Randomly select 1,000 sequences
training_sequences = random.sample(
    dna_sequences,
    1000
)

# Encode the sequences
training_sequences = [
    encode(sequence)
    for sequence in training_sequences
]

generated_data = LSTM.LSTM(training_sequences)

pytorchMMD.mmd_test(torch.tensor(training_sequences), torch.tensor(generated_data))