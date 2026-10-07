from tkinter.filedialog import test

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import stringGenerator as sg
from torch.distributions.categorical import Categorical

def LSTM(inputArray):

    text = torch.tensor(inputArray)
    num_samples = len(inputArray)
    text_encoded = text
    vocab_size = int(torch.max(text).item()) + 1

    seq_length = len(inputArray[0])
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

    text_chunks = []

    for row in text:
        if seq_length > 1:
            text_chunks.append(row)

    seq_dataset = TextDataset(text_chunks)

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

        for c in range(seq_length-1):
            pred, hidden, cell = model(seq_batch[:, c], hidden, cell) 
            loss += loss_fn(pred, target_batch[:, c])

        loss.backward()
        optimizer.step()
        loss = loss.item()/(seq_length-1)
        if epoch % 50 == 0:
            print(f'Epoch {epoch} loss: {loss:.4f}')

    lstm_predictions = []
    for j in range(num_samples):
        sample_seq = torch.randint(0, vocab_size, (1,))

        while len(sample_seq) < seq_length:
            hidden, cell = model.init_hidden(1)
            x = sample_seq[-1].unsqueeze(0)  # shape: (1,)
            logits, hidden, cell = model(x, hidden, cell)

            # print('Probabilities:', nn.functional.softmax(logits, dim=1).detach().numpy()[0])

            # print(sample_seq)
            # print('Samples:')
            m = Categorical(logits=logits)
            samples = m.sample((10,))
            # print(samples.detach().numpy())
            sample_seq = torch.cat([sample_seq, samples[-1].view(-1)])

        lstm_predictions.append(sample_seq.numpy())

    return lstm_predictions