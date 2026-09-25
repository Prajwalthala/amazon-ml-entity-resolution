import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import time

VAL_DIR = Path("student_resource/dataset/val")
TRAIN_DIR = Path("student_resource/dataset/train")

print("Loading validation set...")
val_gt = pd.read_csv(VAL_DIR / "val_ground_truth.tsv", sep="\t")
val_s1 = pd.read_csv(VAL_DIR / "val_source1.tsv", sep="\t")

gt_map = {}
total_gt_pairs = 0
for _, row in val_gt.iterrows():
    s1_id = row["source1_entity_id"]
    m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
    m_set = set(i.strip() for i in m_str.split(",") if i.strip())
    gt_map[s1_id] = m_set
    total_gt_pairs += len(m_set)

print(f"Total ground truth positive pairs: {total_gt_pairs:,}")

print("Loading Source 2 and Source 3...")
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")

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
    "house", "home", "world", "life", "star", "royal", "sun", "moon", "and", "or", "of", "in", "for", "to",
    "com", "net", "org", "www"
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
    distinctive = [t for t in tokens if len(t) >= 2 and t not in GENERIC_WORDS and not t.isdigit()]
    return distinctive

def extract_address_components(addr_clean):
    if not addr_clean:
        return [], []
    nums = re.findall(r"\b\d+[a-z]?\b", addr_clean)
    words = [w for w in addr_clean.split() if len(w) >= 3 and not w.isdigit()]
    return [n for n in nums if len(n) <= 6], words

print("Preprocessing text...")
s2["clean_name"] = s2["business_name"].apply(clean_text)
s2["clean_addr"] = s2["business_address"].apply(clean_text)
s2["dist_tokens"] = s2["clean_name"].apply(extract_distinctive_tokens)

s3["clean_name"] = s3["business_name"].apply(clean_text)
s3["clean_addr"] = s3["business_address"].apply(clean_text)
s3["dist_tokens"] = s3["clean_name"].apply(extract_distinctive_tokens)

val_s1["clean_name"] = val_s1["business_name"].apply(clean_text)
val_s1["clean_addr"] = val_s1["business_address"].apply(clean_text)
val_s1["dist_tokens"] = val_s1["clean_name"].apply(extract_distinctive_tokens)

print("Extracting address components...")
s2["addr_nums_words"] = s2["clean_addr"].apply(extract_address_components)
s3["addr_nums_words"] = s3["clean_addr"].apply(extract_address_components)
val_s1["addr_nums_words"] = val_s1["clean_addr"].apply(extract_address_components)

print("Calculating token frequencies...")
token_freq = defaultdict(int)
for row in s2[["country", "dist_tokens"]].itertuples(index=False):
    c, d_toks = row[0].lower(), row[1]
    for t in set(d_toks):
        token_freq[f"{c}_{t}"] += 1
for row in s3[["country", "dist_tokens"]].itertuples(index=False):
    c, d_toks = row[0].lower(), row[1]
    for t in set(d_toks):
        token_freq[f"{c}_{t}"] += 1

def get_blocking_keys(c, clean_name, d_toks, addr_nums_words):
    keys = []
    a_nums, a_words = addr_nums_words
    
    # 1. Distinctive name tokens (freq <= 2000)
    for t in set(d_toks):
        if token_freq.get(f"{c}_{t}", 999999) <= 2000:
            keys.append(f"{c}_tok_{t}")
            
    # 2. Token pair keys (first 2 sorted distinctive tokens)
    if len(d_toks) >= 2:
        sorted_t = sorted(set(d_toks))
        for i in range(min(3, len(sorted_t))):
            for j in range(i + 1, min(4, len(sorted_t))):
                keys.append(f"{c}_pair_{sorted_t[i]}_{sorted_t[j]}")

    # 3. Concatenated name key without spaces/punct (first 12 chars)
    concat_name = clean_name.replace(" ", "")
    for word in d_toks[:2]:
        if len(word) >= 5:
            keys.append(f"{c}_concat_{word[:8]}")

    # 4. Address number + Address street word key
    if a_nums and a_words:
        for num in a_nums[:2]:
            for word in a_words[:3]:
                keys.append(f"{c}_addr_{num}_{word}")

    # 5. Address number + first 3 letters of first name token
    if a_nums and d_toks:
        t0 = d_toks[0][:3]
        for num in a_nums[:2]:
            keys.append(f"{c}_numt0_{num}_{t0}")

    return set(keys)

print("Building inverted index for S2 and S3...")
index = defaultdict(list)
start_time = time.time()

for row in s2[["entity_id", "country", "clean_name", "dist_tokens", "addr_nums_words"]].itertuples(index=False):
    eid, c, cname, d_toks, a_nw = row[0], row[1].lower(), row[2], row[3], row[4]
    for k in get_blocking_keys(c, cname, d_toks, a_nw):
        index[k].append(eid)

for row in s3[["entity_id", "country", "clean_name", "dist_tokens", "addr_nums_words"]].itertuples(index=False):
    eid, c, cname, d_toks, a_nw = row[0], row[1].lower(), row[2], row[3], row[4]
    for k in get_blocking_keys(c, cname, d_toks, a_nw):
        index[k].append(eid)

print(f"Indexed in {time.time() - start_time:.2f}s. Unique index keys: {len(index):,}")

# Filter hyper-frequent keys from index (freq <= 500)
MAX_KEY_FREQ = 500
active_index = {k: v for k, v in index.items() if len(v) <= MAX_KEY_FREQ}
print(f"Active index keys (freq <= {MAX_KEY_FREQ}): {len(active_index):,}")

print("Querying candidates for Val S1...")
cand_counts = []
recalled_gt_pairs = 0

start_time = time.time()
for row in val_s1[["entity_id", "country", "clean_name", "dist_tokens", "addr_nums_words"]].itertuples(index=False):
    s1_id, c, cname, d_toks, a_nw = row[0], row[1].lower(), row[2], row[3], row[4]
    
    cands = set()
    for k in get_blocking_keys(c, cname, d_toks, a_nw):
        if k in active_index:
            cands.update(active_index[k])

    cand_counts.append(len(cands))
    
    gt_set = gt_map.get(s1_id, set())
    if gt_set:
        recalled_gt_pairs += len(gt_set.intersection(cands))

recall = recalled_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
avg_cands = np.mean(cand_counts)
median_cands = np.median(cand_counts)
p95_cands = np.percentile(cand_counts, 95)

print("\n" + "=" * 60)
print("MULTI-PASS NAME + ADDRESS BLOCKING RESULTS")
print("=" * 60)
print(f"Candidate Generation Time: {time.time() - start_time:.2f}s")
print(f"GT Pair Recall: {recalled_gt_pairs:,} / {total_gt_pairs:,} ({recall * 100:.2f}%)")
print(f"Avg Candidates per S1: {avg_cands:.2f}")
print(f"Median Candidates per S1: {median_cands:.0f}")
print(f"95th Percentile Candidates per S1: {p95_cands:.0f}")
