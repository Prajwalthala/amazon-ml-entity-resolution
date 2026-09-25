import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_DIR = Path("student_resource/dataset/train")

print("Loading ground truth...")
gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")
print(f"Total S1 records in GT: {len(gt):,}")

# Parse matched entity IDs
print("Analyzing match counts...")
gt["matched_list"] = gt["matched_entity_ids"].fillna("").apply(lambda x: [i.strip() for i in x.split(",") if i.strip()])
gt["num_matches"] = gt["matched_list"].apply(len)

num_singletons = (gt["num_matches"] == 0).sum()
num_matched = (gt["num_matches"] > 0).sum()

print(f"Singletons (0 matches): {num_singletons:,} ({num_singletons / len(gt) * 100:.2f}%)")
print(f"Matched S1 entities (>=1 match): {num_matched:,} ({num_matched / len(gt) * 100:.2f}%)")

# Break down S2 vs S3 matches
gt["s2_matches"] = gt["matched_list"].apply(lambda lst: [i for i in lst if i.startswith("S2-")])
gt["s3_matches"] = gt["matched_list"].apply(lambda lst: [i for i in lst if i.startswith("S3-")])

gt["num_s2"] = gt["s2_matches"].apply(len)
gt["num_s3"] = gt["s3_matches"].apply(len)

print("\nMatch breakdown:")
print(f"Has S2 match: {(gt['num_s2'] > 0).sum():,} ({(gt['num_s2'] > 0).sum() / len(gt) * 100:.2f}%)")
print(f"Has S3 match: {(gt['num_s3'] > 0).sum():,} ({(gt['num_s3'] > 0).sum() / len(gt) * 100:.2f}%)")
print(f"Has BOTH S2 and S3 matches: {((gt['num_s2'] > 0) & (gt['num_s3'] > 0)).sum():,}")

total_s2_positive_pairs = gt["num_s2"].sum()
total_s3_positive_pairs = gt["num_s3"].sum()
print(f"Total positive S1-S2 pairs: {total_s2_positive_pairs:,}")
print(f"Total positive S1-S3 pairs: {total_s3_positive_pairs:,}")
print(f"Total positive pairs across S2 and S3: {total_s2_positive_pairs + total_s3_positive_pairs:,}")

# Distribution of number of matches per S1 entity
print("\nDistribution of match counts per S1 entity:")
print(gt["num_matches"].value_counts().head(10))

# Read a sample of S1, S2, S3 to inspect countries and text nulls
print("\nInspecting Source files sample & country distributions...")
s1_sample = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", nrows=100000)
s2_sample = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", nrows=100000)
s3_sample = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", nrows=100000)

print(f"S1 country counts (in 100k sample):")
print(s1_sample["country"].value_counts(dropna=False))
print(f"S2 country counts (in 100k sample):")
print(s2_sample["country"].value_counts(dropna=False))
print(f"S3 country counts (in 100k sample):")
print(s3_sample["country"].value_counts(dropna=False))

print(f"\nS1 null counts:")
print(s1_sample.isnull().sum())
print(f"S2 null counts:")
print(s2_sample.isnull().sum())
print(f"S3 null counts:")
print(s3_sample.isnull().sum())
