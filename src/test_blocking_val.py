import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import time

VAL_DIR = Path("student_resource/dataset/val")
TRAIN_DIR = Path("dataset/train")

print("Loading validation set...")
val_gt = pd.read_csv(VAL_DIR / "val_ground_truth.tsv", sep="\t")
val_s1 = pd.read_csv(VAL_DIR / "val_source1.tsv", sep="\t")

print(f"Validation S1 entities: {len(val_s1):,}")

# Parse ground truth into mapping
gt_map = {}
total_gt_pairs = 0
for _, row in val_gt.iterrows():
    s1_id = row["source1_entity_id"]
    m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
    m_set = set(i.strip() for i in m_str.split(",") if i.strip())
    gt_map[s1_id] = m_set
    total_gt_pairs += len(m_set)

print(f"Total ground truth positive pairs in val split: {total_gt_pairs:,}")

# Load full Source 2 and Source 3 for blocking
print("Loading Source 2 and Source 3...")
start_time = time.time()
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")
print(f"Loaded S2 ({len(s2):,}) and S3 ({len(s3):,}) in {time.time() - start_time:.2f}s")

# Define generic business words to filter out
GENERIC_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited", "corp", "corporation", "company", "co", "plc",
    "pvt", "private", "prvt", "mr", "mrs", "ms", "dr", "shri", "smt", "sri", "shree", "saint", "st",
    "the", "group", "services", "service", "solutions", "enterprises", "enterprise", "business",
    "industries", "industry", "traders", "trading", "store", "stores", "shop", "shops", "center",
    "centre", "mart", "international", "global", "tech", "technologies", "technology", "medical",
    "clinic", "care", "holdings", "ventures", "systems", "system", "consulting", "agency", "management",
    "properties", "capital", "partners", "financial", "logistics", "express", "motors", "auto",
    "electronics", "pharma", "pharmaceuticals", "foods", "food", "textile", "textiles", "garments",
    "clothing", "supermarket", "hardware", "appliances", "agencies", "hospital", "hospitals", "trust",
    "school", "college", "foundry", "distributors", "distributor", "wholesaler", "wholesalers",
    "retail", "retails", "works", "studio", "studios", "labs", "laboratory", "laboratories", "hotel",
    "hotels", "resort", "resorts", "jewellers", "jeweller", "jewelers", "jeweler", "jewels", "jewel",
    "creations", "creation", "fashions", "fashion", "wear", "wears", "garment", "sports", "firm",
    "house", "home", "world", "life", "star", "royal", "sun", "moon", "and", "or", "of", "in", "for", "to"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_distinctive_tokens(name_clean):
    tokens = name_clean.split()
    distinctive = []
    for t in tokens:
        if len(t) >= 2 and t not in GENERIC_WORDS and not t.isdigit():
            distinctive.append(t)
    return distinctive

def extract_address_numbers(addr_clean):
    if not addr_clean:
        return []
    # Find numbers / building codes in address
    nums = re.findall(r"\b\d+[a-z]?\b", addr_clean)
    return [n for n in nums if len(n) <= 6]

print("Extracting blocking features...")
# Preprocess S2 and S3
s2["clean_name"] = s2["business_name"].apply(clean_text)
s2["clean_addr"] = s2["business_address"].apply(clean_text)
s2["dist_tokens"] = s2["clean_name"].apply(extract_distinctive_tokens)
s2["addr_nums"] = s2["clean_addr"].apply(extract_address_numbers)

s3["clean_name"] = s3["business_name"].apply(clean_text)
s3["clean_addr"] = s3["business_address"].apply(clean_text)
s3["dist_tokens"] = s3["clean_name"].apply(extract_distinctive_tokens)
s3["addr_nums"] = s3["clean_addr"].apply(extract_address_numbers)

# Preprocess Val S1
val_s1["clean_name"] = val_s1["business_name"].apply(clean_text)
val_s1["clean_addr"] = val_s1["business_address"].apply(clean_text)
val_s1["dist_tokens"] = val_s1["clean_name"].apply(extract_distinctive_tokens)
val_s1["addr_nums"] = val_s1["clean_addr"].apply(extract_address_numbers)

print("Building inverted index for S2 and S3...")
start_time = time.time()
# Map token -> list of (id, country)
token_to_s23 = defaultdict(list)
num_prefix_to_s23 = defaultdict(list)

MAX_TOKEN_FREQ = 10_000

for row in s2[["entity_id", "country", "dist_tokens", "addr_nums"]].itertuples(index=False):
    eid, c, d_toks, a_nums = row[0], row[1].lower(), row[2], row[3]
    for tok in set(d_toks):
        token_to_s23[f"{c}_{tok}"].append(eid)
    for num in set(a_nums):
        if d_toks:
            prefix = d_toks[0][:3]
            num_prefix_to_s23[f"{c}_{num}_{prefix}"].append(eid)

for row in s3[["entity_id", "country", "dist_tokens", "addr_nums"]].itertuples(index=False):
    eid, c, d_toks, a_nums = row[0], row[1].lower(), row[2], row[3]
    for tok in set(d_toks):
        token_to_s23[f"{c}_{tok}"].append(eid)
    for num in set(a_nums):
        if d_toks:
            prefix = d_toks[0][:3]
            num_prefix_to_s23[f"{c}_{num}_{prefix}"].append(eid)

print(f"Built index in {time.time() - start_time:.2f}s. Unique token keys: {len(token_to_s23):,}")

# Filter hyper-frequent token keys
active_token_to_s23 = {k: v for k, v in token_to_s23.items() if len(v) <= MAX_TOKEN_FREQ}
print(f"Filtered keys with freq <= {MAX_TOKEN_FREQ}: {len(active_token_to_s23):,}")

print("Generating candidates for validation set...")
cand_counts = []
recalled_gt_pairs = 0

start_time = time.time()
for row in val_s1[["entity_id", "country", "dist_tokens", "addr_nums"]].itertuples(index=False):
    s1_id, c, d_toks, a_nums = row[0], row[1].lower(), row[2], row[3]
    
    cands = set()
    # 1. Distinctive token matching
    for tok in set(d_toks):
        key = f"{c}_{tok}"
        if key in active_token_to_s23:
            cands.update(active_token_to_s23[key])
            
    # 2. Number + prefix matching
    for num in set(a_nums):
        if d_toks:
            prefix = d_toks[0][:3]
            key = f"{c}_{num}_{prefix}"
            if key in num_prefix_to_s23:
                cands.update(num_prefix_to_s23[key])

    cand_counts.append(len(cands))
    
    # Measure ground truth recall
    gt_set = gt_map.get(s1_id, set())
    if gt_set:
        recalled_gt_pairs += len(gt_set.intersection(cands))

recall = recalled_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
avg_cands = np.mean(cand_counts)
median_cands = np.median(cand_counts)
p95_cands = np.percentile(cand_counts, 95)

print("\n" + "=" * 60)
print("BLOCKING BENCHMARK RESULTS")
print("=" * 60)
print(f"Candidate Generation Time: {time.time() - start_time:.2f}s for {len(val_s1):,} S1 entities")
print(f"GT Pair Recall: {recalled_gt_pairs:,} / {total_gt_pairs:,} ({recall * 100:.2f}%)")
print(f"Avg Candidates per S1: {avg_cands:.2f}")
print(f"Median Candidates per S1: {median_cands:.0f}")
print(f"95th Percentile Candidates per S1: {p95_cands:.0f}")
