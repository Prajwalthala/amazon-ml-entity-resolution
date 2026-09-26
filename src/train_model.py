import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

from blocking import (
    clean_text,
    extract_name_features,
    extract_address_features,
    generate_blocking_keys,
)

from matching import extract_pair_features, train_model


TRAIN_DIR = Path("student_resource/dataset/train")
VAL_DIR = Path("student_resource/dataset/val")


# Keep training manageable.
# We sample at most this many negative candidates per S1.
MAX_NEGATIVES_PER_S1 = 20


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

start = time.time()

s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)

print(
    f"Loaded data in {time.time() - start:.2f}s"
)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# GROUND TRUTH MAP
# ============================================================

gt_map = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    raw = (
        str(row["matched_entity_ids"])
        if pd.notna(row["matched_entity_ids"])
        else ""
    )

    matches = {
        x.strip()
        for x in raw.split(",")
        if x.strip()
    }

    gt_map[s1_id] = matches


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(df):

    df = df.copy()

    clean_names = df["business_name"].apply(
        clean_text
    )

    clean_addresses = df["business_address"].apply(
        clean_text
    )

    name_features = clean_names.apply(
        extract_name_features
    )

    address_features = clean_addresses.apply(
        extract_address_features
    )

    df["name_toks"] = [
        x[0] for x in name_features
    ]

    df["concat_str"] = [
        x[1] for x in name_features
    ]

    df["prefixes"] = [
        x[2] for x in name_features
    ]

    df["addr_nums"] = [
        x[0] for x in address_features
    ]

    df["addr_words"] = [
        x[1] for x in address_features
    ]

    return df


print("Preprocessing...")

s1 = preprocess(s1)
s2 = preprocess(s2)
s3 = preprocess(s3)


# ============================================================
# COMBINE S2 + S3
# ============================================================

candidates_df = pd.concat(
    [s2, s3],
    ignore_index=True
)

candidate_lookup = (
    candidates_df
    .set_index("entity_id")
    .to_dict(orient="index")
)


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

print("Building blocking index...")

start = time.time()

index = defaultdict(list)

feature_columns = [
    "entity_id",
    "country",
    "name_toks",
    "concat_str",
    "prefixes",
    "addr_nums",
    "addr_words",
]

for row in candidates_df[feature_columns].itertuples(
    index=False
):

    (
        entity_id,
        country,
        name_toks,
        concat_str,
        prefixes,
        addr_nums,
        addr_words,
    ) = row

    keys = generate_blocking_keys(
        country,
        name_toks,
        concat_str,
        prefixes,
        addr_nums,
        addr_words,
    )

    for key in keys:
        index[key].append(entity_id)


MAX_KEY_FREQ = 5000

active_index = {
    key: ids
    for key, ids in index.items()
    if len(ids) <= MAX_KEY_FREQ
}

print(
    f"Blocking index built in "
    f"{time.time() - start:.2f}s"
)

print(
    f"Unique keys: {len(index):,}"
)

print(
    f"Active keys: {len(active_index):,}"
)


# ============================================================
# BUILD TRAINING PAIRS
# ============================================================

print("Generating training pairs...")

X = []
y = []

positive_count = 0
negative_count = 0

rng = np.random.default_rng(42)

start = time.time()


for idx, s1_row in s1.iterrows():

    s1_id = s1_row["entity_id"]

    gt_set = gt_map.get(
        s1_id,
        set()
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    keys = generate_blocking_keys(
        s1_row["country"],
        s1_row["name_toks"],
        s1_row["concat_str"],
        s1_row["prefixes"],
        s1_row["addr_nums"],
        s1_row["addr_words"],
    )

    candidate_ids = set()

    for key in keys:

        if key in active_index:

            candidate_ids.update(
                active_index[key]
            )

    # --------------------------------------------------------
    # Always include known positive pairs
    # --------------------------------------------------------

    candidate_ids.update(gt_set)

    # --------------------------------------------------------
    # Separate positive / negative candidates
    # --------------------------------------------------------

    positives = [
        cid
        for cid in candidate_ids
        if cid in gt_set
    ]

    negatives = [
        cid
        for cid in candidate_ids
        if cid not in gt_set
    ]

    # --------------------------------------------------------
    # Limit negatives
    # --------------------------------------------------------

    if len(negatives) > MAX_NEGATIVES_PER_S1:

        negatives = list(
            rng.choice(
                negatives,
                size=MAX_NEGATIVES_PER_S1,
                replace=False,
            )
        )

    # --------------------------------------------------------
    # Create positive examples
    # --------------------------------------------------------

    for cid in positives:

        candidate = candidate_lookup.get(cid)

        if candidate is None:
            continue

        features = extract_pair_features(
            s1_row,
            candidate
        )

        X.append(features)
        y.append(1)

        positive_count += 1

    # --------------------------------------------------------
    # Create negative examples
    # --------------------------------------------------------

    for cid in negatives:

        candidate = candidate_lookup.get(cid)

        if candidate is None:
            continue

        features = extract_pair_features(
            s1_row,
            candidate
        )

        X.append(features)
        y.append(0)

        negative_count += 1

    if (idx + 1) % 5000 == 0:

        print(
            f"Processed {idx + 1:,}/{len(s1):,} S1"
        )


print(
    f"Training pair generation finished "
    f"in {time.time() - start:.2f}s"
)

print(
    f"Positive pairs: {positive_count:,}"
)

print(
    f"Negative pairs: {negative_count:,}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int8
)

print(
    f"Feature matrix: {X.shape}"
)

print("Training Logistic Regression...")

model = train_model(
    X,
    y
)

print("Training complete.")


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

MODEL_PATH = Path(
    "output/matching_model.pkl"
)

MODEL_PATH.parent.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Model saved to: {MODEL_PATH}"
)
