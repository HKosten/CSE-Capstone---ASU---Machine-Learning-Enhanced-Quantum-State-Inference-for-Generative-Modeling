import numpy as np

P_FIRST_1 = 0.50
P_1_GIVEN_0 = 0.30
P_1_GIVEN_1 = 0.80

SEQ_LENGTH = 8
SAMPLE_SIZES = [100, 1000, 10000, 100000]
ALPHA = 1.0

TRUE_TRANSITIONS = np.array([[1.0 - P_1_GIVEN_0, P_1_GIVEN_0],
                             [1.0 - P_1_GIVEN_1, P_1_GIVEN_1]])

rng = np.random.default_rng(0)


def generate_bitstrings(n_samples, length):
    samples = np.zeros((n_samples, length), dtype=np.int8)
    samples[:, 0] = rng.random(n_samples) < P_FIRST_1
    for i in range(1, length):
        threshold = np.where(samples[:, i - 1] == 1, P_1_GIVEN_1, P_1_GIVEN_0)
        samples[:, i] = rng.random(n_samples) < threshold
    return samples


def count_transitions(samples):
    prev = samples[:, :-1].ravel()
    nxt = samples[:, 1:].ravel()
    counts = np.zeros((2, 2))
    for a in (0, 1):
        for b in (0, 1):
            counts[a][b] = np.sum((prev == a) & (nxt == b))
    return counts


def estimate_transitions(counts, alpha=ALPHA):
    smoothed = counts + alpha
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def sequence_distribution(p_first_1, transitions, length):
    probs = np.zeros(2 ** length)
    for index in range(2 ** length):
        bits = [(index >> (length - 1 - i)) & 1 for i in range(length)]
        p = p_first_1 if bits[0] == 1 else 1.0 - p_first_1
        for i in range(1, length):
            p *= transitions[bits[i - 1]][bits[i]]
        probs[index] = p
    return probs


def total_variation(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def fit(samples):
    transitions = estimate_transitions(count_transitions(samples))
    p_first_1 = samples[:, 0].mean()
    return p_first_1, transitions


true_dist = sequence_distribution(P_FIRST_1, TRUE_TRANSITIONS, SEQ_LENGTH)

print("Ground-truth chain")
print(f"  P(first bit = 1) = {P_FIRST_1:.4f}")
print(f"  P(1 | 0)         = {P_1_GIVEN_0:.4f}")
print(f"  P(1 | 1)         = {P_1_GIVEN_1:.4f}")
print(f"  distribution over {2 ** SEQ_LENGTH} bitstrings of length {SEQ_LENGTH}")

largest = SAMPLE_SIZES[-1]
p_first_hat, transitions_hat = fit(generate_bitstrings(largest, SEQ_LENGTH))

print(f"\nEstimated from {largest} bitstrings")
print(f"  P(first bit = 1) = {p_first_hat:.4f}  (error {abs(p_first_hat - P_FIRST_1):.4f})")
print(f"  P(1 | 0)         = {transitions_hat[0][1]:.4f}  (error {abs(transitions_hat[0][1] - P_1_GIVEN_0):.4f})")
print(f"  P(1 | 1)         = {transitions_hat[1][1]:.4f}  (error {abs(transitions_hat[1][1] - P_1_GIVEN_1):.4f})")

print("\nSample-count sweep")
print(f"{'samples':>10}  {'P(1|0) err':>11}  {'P(1|1) err':>11}  {'TVD':>9}")
for n_samples in SAMPLE_SIZES:
    p_first, transitions = fit(generate_bitstrings(n_samples, SEQ_LENGTH))
    est_dist = sequence_distribution(p_first, transitions, SEQ_LENGTH)
    err_0 = abs(transitions[0][1] - P_1_GIVEN_0)
    err_1 = abs(transitions[1][1] - P_1_GIVEN_1)
    print(f"{n_samples:>10}  {err_0:>11.4f}  {err_1:>11.4f}  {total_variation(true_dist, est_dist):>9.4f}")
