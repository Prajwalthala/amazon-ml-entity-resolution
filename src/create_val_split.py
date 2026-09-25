import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_DIR = Path("student_resource/dataset/train")
VAL_DIR = Path("student_resource/dataset/val")
VAL_DIR.mkdir(exist_ok=True, parents=True)

VAL_SIZE = 50_000

print(f"Sampling {VAL_SIZE:,} S1 entities for validation split...")
gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")

# Stratified sampling: preserve proportion of singletons vs matched
gt["has_match"] = gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "")

val_gt = gt.groupby("has_match", group_keys=False).apply(
    lambda x: x.sample(min(len(x), int(VAL_SIZE * len(x) / len(gt))), random_state=42)
).reset_index(drop=True)

print(f"Validation S1 count: {len(val_gt):,}")
val_s1_ids = set(val_gt["source1_entity_id"])

# Extract matching S2/S3 IDs in GT
val_matched_s2_s3 = set()
for m in val_gt["matched_entity_ids"].dropna():
    if m.strip():
        val_matched_s2_s3.update(i.strip() for i in m.split(","))

print(f"Total true S2/S3 match IDs in validation GT: {len(val_matched_s2_s3):,}")

# Save validation ground truth
val_gt.to_csv(VAL_DIR / "val_ground_truth.tsv", sep="\t", index=False)

# Save validation Source 1
s1_full = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t")
val_s1 = s1_full[s1_full["entity_id"].isin(val_s1_ids)].reset_index(drop=True)
val_s1.to_csv(VAL_DIR / "val_source1.tsv", sep="\t", index=False)

print(f"Validation S1 file saved: {len(val_s1):,} rows.")
