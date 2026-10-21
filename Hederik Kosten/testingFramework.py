import random

import LSTM
import pytorchMMD
import torch
import random
import Qubit_String_Generator as qsg

random.seed(0)
with open("chr1.fa", 'r') as f:
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

with open("training_sequences_DNA.csv", 'w') as f:
    for sequence in training_sequences:
        f.write(','.join(map(str, sequence)) + '\n')

def generate_dna_sequences():
    bases = ['A', 'C', 'G', 'T']

    return [
        ''.join(random.choice(bases) for _ in range(50))
        for _ in range(1000)
    ]

random_dna_sequences = generate_dna_sequences()
# print("Random DNA sequences:", random_dna_sequences[0:5])

random_dna_sequences = [
    encode(sequence)
    for sequence in random_dna_sequences
]

generated_data = LSTM.LSTM("training_sequences_DNA.csv", num_epochs=1000, seed=0, epsilon=0.5, num_samples=1000)

print("Generated DNA vs Random DNA:")
pytorchMMD.mmd_test(torch.tensor(generated_data), torch.tensor(random_dna_sequences))

print("Training DNA vs Generated DNA:")
pytorchMMD.mmd_test(torch.tensor(training_sequences), torch.tensor(generated_data))


input_array = qsg.generate_qubit_samples(1000, 12)

with open("training_sequences_qubit.csv", 'w') as f:
    for sequence in input_array:
        f.write(','.join(map(str, sequence)) + '\n')

alternate_qubit_samples = qsg.generate_qubit_samples(1000, 12, decay=2)

lstm_generated_data = LSTM.LSTM("training_sequences_qubit.csv", num_epochs=1000, seed=0, epsilon=0.5, num_samples=1000)

print("Generated Qubit vs Alternate Qubit:")
pytorchMMD.mmd_test(torch.tensor(lstm_generated_data), torch.tensor(alternate_qubit_samples))

print("Training Qubit vs Generated Qubit:")
pytorchMMD.mmd_test(torch.tensor(input_array.tolist()), torch.tensor(lstm_generated_data))