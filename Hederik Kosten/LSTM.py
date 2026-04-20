from tkinter.filedialog import test

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import stringGenerator as sg
from torch.distributions.categorical import Categorical
import mmd

with open(f"LSTM_predictions.txt", 'w') as f:
    f.write("")

with open(f"string_generator_samples.txt", 'w') as f:
    f.write("")

p1 = 0.9
p2 = 0.1

text = sg.generate_string(p1, p2, 1000)
text_encoded = text
chars = set(text)

seq_length = 100
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size]
for i in range(len(text_encoded)-chunk_size)]
    
class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()

seq_dataset = TextDataset(torch.tensor(text_chunks))

batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out.squeeze(1))
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
    
char_array = sorted(list(chars))
vocab_size = len(char_array)
embed_dim = 8
rnn_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
torch.manual_seed(1)
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    hidden = hidden.detach()
    cell = cell.detach()
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0

    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell) 
        loss += loss_fn(pred, target_batch[:, c])

    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 50 == 0:
        print(f'Epoch {epoch} loss: {loss:.4f}')

for j in range(100):
    sample_seq = torch.tensor(sg.generate_string(p1, p2, seq_length)[:seq_length])

    for i in range(12):
        hidden, cell = model.init_hidden(1)
        for c in range(i, len(sample_seq)):
            x = sample_seq[c].unsqueeze(0)  # shape: (1,)
            logits, hidden, cell = model(x, hidden, cell)

        # print('Probabilities:', nn.functional.softmax(logits, dim=1).detach().numpy()[0])

        # print(sample_seq)
        # print('Samples:')
        m = Categorical(logits=logits)
        samples = m.sample((10,))
        # print(samples.detach().numpy())
        sample_seq = torch.cat([sample_seq, samples[-1].view(-1)])

    with open("LSTM_predictions.txt", 'a') as f:
        f.write(str(sample_seq[-12:].numpy()) + '\n')

for i in range(100):
    with open(f"string_generator_samples.txt", 'a') as f:
        f.write(str(sg.generate_string(p1, p2, 12)) + "\n")

with open(f"LSTM_predictions.txt", 'r') as f:
    lstm_predictions = [list(map(int, line.strip()[1:-1].split())) for line in f.readlines()]

with open(f"string_generator_samples.txt", 'r') as f:
    samples = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

print(f"MMD (Gaussian Kernel): {mmd.mmd(samples, lstm_predictions, 'gaussian')}")
print(f"MMD (Spectrum Kernel): {mmd.mmd(samples, lstm_predictions, 'spectrum')}")