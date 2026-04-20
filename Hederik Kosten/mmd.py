import torch

def mmd(x, y, kernel):
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0

    sigma = median_pairwise_distance(x + y) ** 0.5
    # print(f"Median pairwise distance: {sigma}")

    for i in range(len(x)):
        for j in range(len(x)):
            if kernel == 'gaussian':
                k_xx = gaussian_kernel(x[i], x[j], sigma)
            else:
                k_xx = spectrum_kernel(x[i], x[j])
            sum_xx += k_xx

    for i in range(len(y)):
        for j in range(len(y)):
            if kernel == 'gaussian':
                k_yy = gaussian_kernel(y[i], y[j], sigma)
            else:
                k_yy = spectrum_kernel(y[i], y[j])
            sum_yy += k_yy

    for i in range(len(x)):
        for j in range(len(y)):
            if kernel == 'gaussian':
                k_xy = gaussian_kernel(x[i], y[j], sigma)
            else:
                k_xy = spectrum_kernel(x[i], y[j])
            sum_xy += k_xy

    return sum_xx / (len(x) * len(x)) + sum_yy / (len(y) * len(y)) - 2 * sum_xy / (len(x) * len(y))

def gaussian_kernel(x, y, sigma=3.0):
    sum_array = 0.0
    for i in range(len(x)):
        sum_array += (x[i] - y[i]) ** 2
    return (torch.exp(torch.tensor(-sum_array / (2 * sigma ** 2)))).item()

def median_pairwise_distance(x):
    distances = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):  # i+1 to avoid duplicates and self-pairs
            dist = sum((x[i][k] - x[j][k]) ** 2 for k in range(len(x[i])))
            distances.append(dist)
    return torch.median(torch.tensor(distances, dtype=torch.float32)).item()

def get_bigrams(sample):
    bigrams = {}
    for i in range(len(sample) - 1):
        pair = (sample[i], sample[i+1])
        if pair in bigrams:
            bigrams[pair] += 1
        else:
            bigrams[pair] = 1
    return bigrams

def spectrum_kernel(s1, s2):
    bigrams1 = get_bigrams(s1)
    bigrams2 = get_bigrams(s2)
    
    dot_product = 0
    for pair in bigrams1:
        if pair in bigrams2:
            dot_product += bigrams1[pair] * bigrams2[pair]
    return dot_product