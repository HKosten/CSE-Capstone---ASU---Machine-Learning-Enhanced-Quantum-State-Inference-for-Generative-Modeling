import LSTM
import stringGenerator as sg
import pytorchMMD
import torch

training_data = sg.generate_string(0.9, 0.1, 1200)
generated_data = LSTM.LSTM(training_data)

training_sequences = [
    training_data[i:i + 12]
    for i in range(0, 1200, 12)
]

pytorchMMD.mmd_test(torch.tensor(training_sequences), torch.tensor(generated_data))