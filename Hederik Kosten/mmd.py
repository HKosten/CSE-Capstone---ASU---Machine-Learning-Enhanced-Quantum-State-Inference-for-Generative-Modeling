import torch

def mmd(x, y):
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0

    sigma = median_pairwise_distance(x + y) ** 0.5
    print(f"Median pairwise distance: {sigma}")

    for i in range(len(x)):
        for j in range(len(x)):
            k_xx = gaussian_kernel(x[i], x[j], sigma)
            sum_xx += k_xx

    for i in range(len(y)):
        for j in range(len(y)):
            k_yy = gaussian_kernel(y[i], y[j], sigma)
            sum_yy += k_yy

    for i in range(len(x)):
        for j in range(len(y)):
            k_xy = gaussian_kernel(x[i], y[j], sigma)
            sum_xy += k_xy

    return sum_xx / (len(x) * len(x)) + sum_yy / (len(y) * len(y)) - 2 * sum_xy / (len(x) * len(y))

def gaussian_kernel(x, y, sigma=3.0):
    sum_array = 0.0
    for i in range(len(x)):
        sum_array += (x[i] - y[i]) ** 2
    return torch.exp(torch.tensor(-sum_array / (2 * sigma ** 2)))

def median_pairwise_distance(x):
    distances = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):  # i+1 to avoid duplicates and self-pairs
            dist = sum((x[i][k] - x[j][k]) ** 2 for k in range(len(x[i])))
            distances.append(dist)
    return torch.median(torch.tensor(distances, dtype=torch.float32)).item()