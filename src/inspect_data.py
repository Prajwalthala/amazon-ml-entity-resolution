import pandas as pd
from pathlib import Path


TRAIN_DIR = Path("student_resource/dataset/train")


def inspect_file(filename):
    path = TRAIN_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    # Only read the first 5 rows
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5
    )

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.to_string(index=False))


files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    inspect_file(file)