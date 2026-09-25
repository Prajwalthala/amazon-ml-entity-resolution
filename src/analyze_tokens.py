import pandas as pd
from collections import Counter
from pathlib import Path

from normalize import normalize_name, normalize_address


TRAIN_DIR = Path("student_resource/dataset/train")

SOURCES = [
    TRAIN_DIR / "train_source1.tsv",
    TRAIN_DIR / "train_source2.tsv",
    TRAIN_DIR / "train_source3.tsv",
]


for source_file in SOURCES:

    print(f"\nProcessing {source_file.name}...")

    name_counter = Counter()
    address_counter = Counter()

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        usecols=[
            "business_name",
            "business_address"
        ],
        chunksize=100_000
    ):

        for name in chunk["business_name"].fillna(""):

            normalized = normalize_name(name)

            tokens = set(normalized.split())

            name_counter.update(tokens)

        for address in chunk["business_address"].fillna(""):

            normalized = normalize_address(address)

            tokens = set(normalized.split())

            address_counter.update(tokens)

    print("\nMost common NAME tokens:")

    for token, count in name_counter.most_common(30):
        print(f"{token:30} {count}")

    print("\nMost common ADDRESS tokens:")

    for token, count in address_counter.most_common(30):
        print(f"{token:30} {count}")