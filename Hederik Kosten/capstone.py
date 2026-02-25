import torch
import torch.nn as nn
import torch.optim as optim
import random
class CoinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))  # θ

    def forward(self):
        return torch.sigmoid(self.logit)  # p in (0,1)

coinResultList = []
for i in range(0, 1000):
    coinResultList.append(random.randint(0, 1))
print(coinResultList)

y = torch.tensor(coinResultList, dtype=torch.float32)
criterion = nn.BCELoss()

model = CoinModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    optimizer.zero_grad()

    p = model()                  # predicted probability
    loss = criterion(p.expand_as(y), y)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | p = {p.item():.4f} | loss = {loss.item():.4f}")

print("Estimated probability of heads:", model().item())
print("True empirical probability:", sum(coinResultList)/len(coinResultList))
