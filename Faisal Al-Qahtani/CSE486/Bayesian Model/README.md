import csv
import random

SEED = 50
ROWS = 8
COLS = 16
NUM_SAMPLES = 1000
OUTPUT_FILE = "bars_stripes_128.csv"

random.seed(SEED)


def horizontal_stripes():
    """Create an 8x16 binary grid with each row entirely 0s or 1s."""
    grid = []
    for _ in range(ROWS):
        row_value = random.randint(0, 1)
        grid.extend([row_value] * COLS)
    return grid


def vertical_bars():
    """Create an 8x16 binary grid with each column entirely 0s or 1s."""
    columns = [random.randint(0, 1) for _ in range(COLS)]
    grid = []
    for _ in range(ROWS):
        grid.extend(columns)
    return grid


def generate_unique_dataset(num_samples=NUM_SAMPLES):
    samples = set()

    while len(samples) < num_samples:
        if random.random() < 0.5:
            sample = horizontal_stripes()
        else:
            sample = vertical_bars()

        bitstring = "".join(str(bit) for bit in sample)
        samples.add(bitstring)

    return sorted(samples)


def save_dataset(samples, filename=OUTPUT_FILE):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bitstring"])
        for bitstring in samples:
            writer.writerow([bitstring])


if __name__ == "__main__":
    dataset = generate_unique_dataset()
    save_dataset(dataset)

    print(f"Generated {len(dataset)} unique samples.")
    print(f"Each sample contains {ROWS * COLS} bits.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFirst 5 samples:")
    for sample in dataset[:5]:
        print(sample)
