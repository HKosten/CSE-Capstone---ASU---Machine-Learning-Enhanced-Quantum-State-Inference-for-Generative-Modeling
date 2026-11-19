import pandas as pd
from pathlib import Path

folder = Path("Generated Data")

for file in folder.iterdir():
    if file.is_file() and "generated_data_" in file.name:
        df = pd.read_csv(file, header=None)
        df = df.astype(int)

        new_file = file.with_name(file.name + ".csv")
        df.to_csv(new_file, index=False, header=False)

        file.unlink()

        print(f"Cleaned: {new_file.name}")

print("Finished cleaning generated data!")