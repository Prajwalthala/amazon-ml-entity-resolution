import pandas as pd
from pathlib import Path

from normalize import normalize_name, normalize_address
from blocking import add_blocking_keys


TRAIN_DIR = Path("student_resource/dataset/train")

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"


def prepare_chunk(df):

    df["name_normalized"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_name)
    )

    df["address_normalized"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )

    df = add_blocking_keys(df)

    return df


print("Loading Source 1...")

source1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

source1 = prepare_chunk(source1)

print("\nSource 1 loaded:")
print(source1.head())

print("\nNumber of Source 1 records:")
print(len(source1))


# ---------------------------------
# Count Source 1 blocking keys
# ---------------------------------

print("\nCreating Source 1 block statistics...")

source1_key_counts = {}

for keys in source1["blocking_keys"]:

    for key in keys:

        source1_key_counts[key] = (
            source1_key_counts.get(key, 0) + 1
        )


source1_blocks = pd.Series(source1_key_counts)


print("\nNumber of Source 1 blocking keys:")
print(len(source1_blocks))

print("\nLargest Source 1 blocks:")

print(
    source1_blocks
    .sort_values(ascending=False)
    .head(20)
)


# ---------------------------------
# Process Source 2
# ---------------------------------

print("\nProcessing Source 2 in chunks...")

source2_key_counts = {}

for chunk in pd.read_csv(
    SOURCE2,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    chunksize=100_000
):

    chunk = prepare_chunk(chunk)

    for keys in chunk["blocking_keys"]:

        for key in keys:

            source2_key_counts[key] = (
                source2_key_counts.get(key, 0) + 1
            )


source2_blocks = pd.Series(source2_key_counts)


print("\nNumber of Source 2 blocking keys:")
print(len(source2_blocks))

print("\nLargest Source 2 blocks:")

print(
    source2_blocks
    .sort_values(ascending=False)
    .head(20)
)