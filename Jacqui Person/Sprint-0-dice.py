import torch
import torch.nn as nn
import torch.optim as optim

#training data
num_samples = 10000
rolls = torch.randint(0, 6, (num_samples,)) 

#logits - raw, unormalized scores nn outputs
#softmax - converts into probability distribution
class DiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(6))  

    def forward(self):
        return torch.softmax(self.logits, dim=0)

#optimizer - updates model's params to reduce loss 
#(difference between ground truth & expected outcome) 
#
model = DiceModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

#passes through data 200 times
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    
    logits = model.logits.unsqueeze(0).repeat(num_samples, 1)
    loss = criterion(logits, rolls)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    probs = torch.softmax(model.logits, dim=0)
    print("Learned probabilities:", probs)

