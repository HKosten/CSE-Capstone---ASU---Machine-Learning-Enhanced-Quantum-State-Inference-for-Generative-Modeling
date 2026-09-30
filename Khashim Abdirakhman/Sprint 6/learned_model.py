import numpy as np

L, K = 8, 2                      # sequence length, how many past bits the model sees
rng = np.random.default_rng(0)

def gen(n, rule):                # rule(prev1, prev2) -> P(next=1)
    s = np.zeros((n, L), dtype=int)
    s[:, 0] = rng.random(n) < 0.5
    s[:, 1] = rng.random(n) < rule(s[:, 0], np.zeros(n, int))
    for i in range(2, L):
        s[:, i] = rng.random(n) < rule(s[:, i-1], s[:, i-2])
    return s

rule_1mem = lambda a, b: np.where(a == 1, 0.8, 0.3)                      # same as baseline
rule_2mem = lambda a, b: np.array([0.1, 0.9, 0.7, 0.2])[2*a + b]         # depends on last 2 bits

def context(s, i):               # one-hot of the last K bits (4 options) + position flag
    a = s[:, i-1] if i >= 1 else np.zeros(len(s), int)
    b = s[:, i-2] if i >= 2 else np.zeros(len(s), int)
    x = np.zeros((len(s), 4)); x[np.arange(len(s)), 2*a + b] = 1
    return x

def train(s, steps=2000, lr=1.0):
    W = np.zeros((L, 4))         # one small logistic model per position
    for i in range(L):
        x, y = context(s, i), s[:, i]
        for _ in range(steps):
            p = 1 / (1 + np.exp(-x @ W[i]))
            W[i] -= lr * x.T @ (p - y) / len(y)
    return W

def model_dist(W):
    out = np.zeros(2**L)
    for idx in range(2**L):
        bits = np.array([[(idx >> (L-1-i)) & 1 for i in range(L)]])
        p = 1.0
        for i in range(L):
            q = 1 / (1 + np.exp(-context(bits, i) @ W[i]))[0]
            p *= q if bits[0, i] else 1 - q
        out[idx] = p
    return out

def true_dist(rule):
    big = gen(1, rule)  # dummy to reuse shape
    out = np.zeros(2**L)
    for idx in range(2**L):
        b = [(idx >> (L-1-i)) & 1 for i in range(L)]
        p = 0.5
        for i in range(1, L):
            q = rule(np.array(b[i-1]), np.array(b[i-2] if i >= 2 else 0))
            p *= q if b[i] else 1 - q
        out[idx] = p
    return out

def counting_dist(s):            # the baseline: 1-bit memory counting
    prev, nxt = s[:, :-1].ravel(), s[:, 1:].ravel()
    T = np.array([[np.sum((prev==a)&(nxt==c))+1 for c in (0,1)] for a in (0,1)], float)
    T /= T.sum(1, keepdims=True); p1 = s[:, 0].mean()
    out = np.zeros(2**L)
    for idx in range(2**L):
        b = [(idx >> (L-1-i)) & 1 for i in range(L)]
        p = p1 if b[0] else 1 - p1
        for i in range(1, L): p *= T[b[i-1], b[i]]
        out[idx] = p
    return out

tvd = lambda p, q: 0.5 * np.abs(p - q).sum()

for name, rule in [("Test 1: 1-bit memory", rule_1mem), ("Test 2: 2-bit memory", rule_2mem)]:
    truth = true_dist(rule)
    print(f"\n{name}")
    print(f"{'samples':>8} | {'counting TVD':>12} | {'learned TVD':>11}")
    for n in [100, 1000, 10000]:
        s = gen(n, rule)
        print(f"{n:>8} | {tvd(truth, counting_dist(s)):>12.4f} | {tvd(truth, model_dist(train(s))):>11.4f}")
