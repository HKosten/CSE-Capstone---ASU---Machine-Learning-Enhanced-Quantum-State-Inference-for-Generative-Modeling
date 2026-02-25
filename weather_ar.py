# Simple autoregressive (AR(1)) weather model in PyTorch
# Learns P(x_t | x_{t-1}) for two states: Sunny=0, Rainy=1

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# ----------------------------
# 1) True transition probs
# ----------------------------
# P(next=Sunny | prev=Sunny) = 0.9  => P(next=Rainy | prev=Sunny) = 0.1
# P(next=Sunny | prev=Rainy) = 0.2  => P(next=Rainy | prev=Rainy) = 0.8
P_true = torch.tensor([
    [0.9, 0.1],  # prev Sunny -> [Sunny, Rainy]
    [0.2, 0.8],  # prev Rainy -> [Sunny, Rainy]
], dtype=torch.float32)

# ----------------------------
# 2) Generate dependent sequences
# ----------------------------
def generate_weather_sequences(num_seqs=2000, seq_len=30, p_true=P_true, p0=0.5):
    """
    Returns:
      prev_states: (N,) 0/1
      next_states: (N,) 0/1
    where N = num_seqs*(seq_len-1)
    """
    prev_list, next_list = [], []
    for _ in range(num_seqs):
        x = torch.zeros(seq_len, dtype=torch.long)
        # initial day
        x[0] = torch.bernoulli(torch.tensor([1 - p0])).long().item()  # Rainy with prob p0
        # transitions
        for t in range(1, seq_len):
            prev = x[t-1].item()
            probs = p_true[prev]
            x[t] = torch.multinomial(probs, num_samples=1).item()
        prev_list.append(x[:-1])
        next_list.append(x[1:])
    prev_states = torch.cat(prev_list)  # (N,)
    next_states = torch.cat(next_list)  # (N,)
    return prev_states, next_states

prev_states, next_states = generate_weather_sequences()

# One-hot encode previous state as input (autoregressive: use x_{t-1} to predict x_t)
X = torch.nn.functional.one_hot(prev_states, num_classes=2).float()  # (N,2)
y = next_states  # (N,)

# ----------------------------
# 3) Model: explicit conditional probabilities
# ----------------------------
class SimpleARWeather(nn.Module):
    """
    p(next | prev) = softmax(W * onehot(prev) + b)
    This is a 2x2 transition model learned by gradient descent.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True)

    def forward(self, x_onehot):
        logits = self.linear(x_onehot)              # (batch,2)
        probs = torch.softmax(logits, dim=-1)       # explicit probabilities
        return probs, logits

model = SimpleARWeather()

# ----------------------------
# 4) Train with cross-entropy
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

batch_size = 512
num_epochs = 20

for epoch in range(num_epochs):
    perm = torch.randperm(X.size(0))
    X_shuf, y_shuf = X[perm], y[perm]

    total_loss = 0.0
    for i in range(0, X.size(0), batch_size):
        xb = X_shuf[i:i+batch_size]
        yb = y_shuf[i:i+batch_size]

        probs, logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / X.size(0)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d} | loss = {avg_loss:.4f}")

# ----------------------------
# 5) Extract learned conditional probabilities
# ----------------------------
with torch.no_grad():
    eye = torch.eye(2)  # one-hot for [Sunny, Rainy] as prev state
    learned_probs, _ = model(eye)  # rows correspond to prev=Sunny(0), prev=Rainy(1)

print("\nTrue transition probabilities P_true (rows=prev, cols=next [Sunny,Rainy]):")
print(P_true)

print("\nLearned transition probabilities P_learned (rows=prev, cols=next [Sunny,Rainy]):")
print(learned_probs)

print("\nAbsolute error |P_learned - P_true|:")
print((learned_probs - P_true).abs())

# ----------------------------
# 6) Human-readable display
# ----------------------------
states = ["Sunny", "Rainy"]
print("\nReadable comparison:")
for prev in [0, 1]:
    for nxt in [0, 1]:
        print(
            f"P({states[nxt]} | {states[prev]}) true={P_true[prev, nxt]:.3f} "
            f"learned={learned_probs[prev, nxt].item():.3f}"
        )