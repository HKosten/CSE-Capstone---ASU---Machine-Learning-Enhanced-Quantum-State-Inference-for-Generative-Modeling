import numpy as np
import torch
import torch.nn as nn
import random

# ── Config ────────────────────────────────────────────────
STRING_LENGTH  = 100
SAMPLES_EACH   = 200     # training strings per class
EPOCHS         = 20
LR             = 1e-3

# 4 fixed (y1, y2) classes
CLASSES = [(0.9, 0.9), (0.9, 0.6), (0.6, 0.9), (0.6, 0.6)]


# ── Helper: extract consecutive pairs ────────────────────
def extract_pairs(s):
    """[s1,s2,s3,s4] → [[s1,s2],[s2,s3],[s3,s4]]"""
    return [[s[i], s[i+1]] for i in range(len(s) - 1)]


# ── Generate a Markov bit string ──────────────────────────
def make_string(length, y1, y2):
    bits = [random.randint(0, 1)]
    for _ in range(length - 1):
        if bits[-1] == 1:
            bits.append(1 if random.random() < y1 else 0)
        else:
            bits.append(0 if random.random() < y2 else 1)
    return bits


# ── Convert a string to a model-ready tensor ─────────────
def to_tensor(s):
    pairs = extract_pairs(s)                    # (L-1) x 2
    arr = np.array(pairs, dtype=np.float32).T   # 2 x (L-1)
    return torch.tensor(arr)


# ── Build training data as plain arrays ──────────────────
X, y = [], []
for label, (y1, y2) in enumerate(CLASSES):
    for _ in range(SAMPLES_EACH):
        X.append(to_tensor(make_string(STRING_LENGTH, y1, y2)))
        y.append(label)

X = torch.stack(X)                             # (N, 2, L-1)
y = torch.tensor(y)

# Shuffle
perm = torch.randperm(len(y))
X, y = X[perm], y[perm]

# Train / val split (85/15)
split = int(0.85 * len(y))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]


# ── CNN Model ─────────────────────────────────────────────
class MarkovCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Linear(64, len(CLASSES)),
        )
    def forward(self, x):
        return self.net(x)


model    = MarkovCNN()
opt      = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn  = nn.CrossEntropyLoss()


# ── Training loop ─────────────────────────────────────────
print("Training...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    logits = model(X_train)
    loss   = loss_fn(logits, y_train)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={loss.item():.4f}  val_acc={val_acc:.3f}")


# ── Classify a new string ─────────────────────────────────
def classify(bit_string):
    model.eval()
    x = to_tensor(bit_string).unsqueeze(0)      # (1, 2, L-1)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze()
    best  = probs.argmax().item()
    y1, y2 = CLASSES[best]
    print(f"\nPredicted class: (y1={y1}, y2={y2})  confidence={probs[best]:.3f}")
    print("All probabilities:")
    for i, (c1, c2) in enumerate(CLASSES):
        print(f"  (y1={c1}, y2={c2}): {probs[i]:.3f}")
    return CLASSES[best]


# ── Demo ──────────────────────────────────────────────────
print("\n── Test Predictions ─────────────────────────────")
for true_y1, true_y2 in CLASSES:
    test = make_string(STRING_LENGTH, true_y1, true_y2)
    print(f"\nTrue: (y1={true_y1}, y2={true_y2})")
    classify(test)
