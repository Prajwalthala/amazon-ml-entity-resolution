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

print(f"Total ground truth positive pairs in val split: {total_gt_pairs:,}")

print("Loading Source 2 and Source 3...")
start_time = time.time()
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")
print(f"Loaded S2 ({len(s2):,}) and S3 ({len(s3):,}) in {time.time() - start_time:.2f}s")

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

GENERIC_ADDR_WORDS = {
    "road", "street", "drive", "avenue", "lane", "boulevard", "highway", "court",
    "floor", "flat", "unit", "apartment", "no", "number", "rd", "st", "dr", "ave", "ln", "blvd", "hwy", "ct"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_features(b_name, b_addr):
    name_clean = clean_text(b_name)
    addr_clean = clean_text(b_addr)
    
    # 1. Name tokens
    raw_toks = name_clean.split()
    tokens = []
    for t in raw_toks:
        t_stripped = re.sub(r"\.(com|net|org|in|us|uk|co|gov|edu|io|info)$", "", t)
        if len(t_stripped) >= 2 and t_stripped not in GENERIC_WORDS and not t_stripped.isdigit():
            tokens.append(t_stripped)
            
    concat_str = "".join(tokens)
    
    prefixes = set()
    if len(concat_str) >= 5:
        prefixes.add(concat_str[:5])
        prefixes.add(concat_str[:6])
    for t in tokens:
        if len(t) >= 5:
            prefixes.add(t[:5])
            
    # 2. Address features
    addr_nums = []
    if addr_clean:
        raw_nums = set(re.findall(r"\b\d+[a-z]?\b", addr_clean))
        addr_nums = [n for n in raw_nums if len(n) <= 6]
        
    addr_words = []
    if addr_clean:
        addr_words = [w for w in addr_clean.split() if len(w) >= 3 and not w.isdigit() and w not in GENERIC_ADDR_WORDS]
        
    return tokens, concat_str, list(prefixes), addr_nums, addr_words[:6]

print("Extracting features fast...")
def get_dataset_features(df):
    feats = []
    names = df["business_name"].values
    addrs = df["business_address"].values
    for n, a in zip(names, addrs):
        feats.append(extract_features(n, a))
    return feats

s2_feats = get_dataset_features(s2)
s3_feats = get_dataset_features(s3)
val_feats = get_dataset_features(val_s1)

def generate_blocking_keys(country, feat):
    n_toks, c_str, p_fx, a_nums, a_words = feat
    keys = set()
    c = str(country).lower()
    
    # Name tokens
    for t in set(n_toks):
        keys.add(f"{c}_tok_{t}")
        
    # Name pairs
    if len(n_toks) >= 2:
        sorted_t = sorted(set(n_toks))
        for i in range(min(4, len(sorted_t))):
            for j in range(i + 1, min(5, len(sorted_t))):
                keys.add(f"{c}_pair_{sorted_t[i]}_{sorted_t[j]}")
                
    # Concatenated name
    if len(c_str) >= 5:
        keys.add(f"{c}_concat_{c_str}")
        
    # Prefixes
    for p in set(p_fx):
        keys.add(f"{c}_p5_{p}")
        
    # Address Number + Address Word
    if a_nums and a_words:
        for num in a_nums[:2]:
            for aword in a_words[:4]:
                keys.add(f"{c}_anw_{num}_{aword}")
                
    # Address Number + First 3 chars of name token
    if a_nums and n_toks:
        t0 = n_toks[0][:3]
        for num in a_nums[:2]:
            keys.add(f"{c}_anumt_{num}_{t0}")
            
    # Pure Address Pair (Two distinct address words if length >= 4)
    distinct_awords = sorted(list(set(w for w in a_words if len(w) >= 4)))
    if len(distinct_awords) >= 2:
        keys.add(f"{c}_apair_{distinct_awords[0]}_{distinct_awords[1]}")
        
    return keys

print("Building inverted index...")
start_time = time.time()
index = defaultdict(list)

s2_eids = s2["entity_id"].values
s2_ctry = s2["country"].values
for eid, c, feat in zip(s2_eids, s2_ctry, s2_feats):
    for k in generate_blocking_keys(c, feat):
        index[k].append(eid)

s3_eids = s3["entity_id"].values
s3_ctry = s3["country"].values
for eid, c, feat in zip(s3_eids, s3_ctry, s3_feats):
    for k in generate_blocking_keys(c, feat):
        index[k].append(eid)

print(f"Index built in {time.time() - start_time:.2f}s. Unique keys: {len(index):,}")

# Filter max freq = 10000
MAX_KEY_FREQ = 10000
active_index = {k: v for k, v in index.items() if len(v) <= MAX_KEY_FREQ}
print(f"Active index keys (freq <= {MAX_KEY_FREQ}): {len(active_index):,}")

print("Querying candidates for Val S1...")
cand_counts = []
recalled_gt_pairs = 0

val_eids = val_s1["entity_id"].values
val_ctry = val_s1["country"].values

start_time = time.time()
for s1_id, c, feat in zip(val_eids, val_ctry, val_feats):
    cands = set()
    for k in generate_blocking_keys(c, feat):
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
print("FINAL RECALL BLOCKING BENCHMARK RESULTS")
print("=" * 60)
print(f"Candidate Generation Time: {time.time() - start_time:.2f}s for {len(val_s1):,} S1 entities")
print(f"GT Pair Recall: {recalled_gt_pairs:,} / {total_gt_pairs:,} ({recall * 100:.2f}%)")
print(f"Avg Candidates per S1: {avg_cands:.2f}")
print(f"Median Candidates per S1: {median_cands:.0f}")
print(f"95th Percentile Candidates per S1: {p95_cands:.0f}")
