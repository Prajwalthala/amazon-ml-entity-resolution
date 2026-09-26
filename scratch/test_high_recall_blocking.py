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

def extract_name_features(name_clean):
    if not name_clean:
        return [], "", []
    
    # Strip domain suffixes / web prefixes
    raw_tokens = name_clean.split()
    tokens = []
    for t in raw_tokens:
        t_clean = re.sub(r"\.(com|net|org|in|us|uk|co|gov|edu|io|info)$", "", t)
        if len(t_clean) >= 2 and t_clean not in GENERIC_WORDS and not t_clean.isdigit():
            tokens.append(t_clean)
            
    # Concatenated string
    concat_str = "".join(tokens)
    
    # Prefixes (5 chars) of long tokens or concatenated name
    prefixes = []
    if len(concat_str) >= 5:
        prefixes.append(concat_str[:5])
        prefixes.append(concat_str[:6])
    for t in tokens:
        if len(t) >= 6:
            prefixes.append(t[:5])
            
    return tokens, concat_str, list(set(prefixes))

def extract_address_features(addr_clean):
    if not addr_clean:
        return [], []
    nums = set(re.findall(r"\b\d+[a-z]?\b", addr_clean))
    nums = [n for n in nums if len(n) <= 6]
    
    words = [w for w in addr_clean.split() if len(w) >= 3 and not w.isdigit() and w not in GENERIC_ADDR_WORDS]
    return nums, words[:6]

def generate_blocking_keys(country, name_toks, concat_str, prefixes, addr_nums, addr_words):
    keys = set()
    c = country.lower()
    
    # 1. Distinctive name tokens
    for t in set(name_toks):
        keys.add(f"{c}_tok_{t}")
        
    # 2. Token pair keys (first 3 distinctive tokens)
    if len(name_toks) >= 2:
        sorted_t = sorted(set(name_toks))
        for i in range(min(4, len(sorted_t))):
            for j in range(i + 1, min(5, len(sorted_t))):
                keys.add(f"{c}_pair_{sorted_t[i]}_{sorted_t[j]}")
                
    # 3. Concatenated name key
    if len(concat_str) >= 5:
        keys.add(f"{c}_concat_{concat_str}")
        
    # 4. Prefixes (5-gram prefixes)
    for p in set(prefixes):
        keys.add(f"{c}_p5_{p}")
        
        # 5. Address Number + Address Word keys
    if addr_nums and addr_words:
        for num in addr_nums[:2]:
            for aword in addr_words[:3]:
                keys.add(f"{c}_anw_{num}_{aword}")

    # 6. Address word pair keys
    if len(addr_words) >= 2:
        unique_words = list(dict.fromkeys(addr_words))
        for i in range(min(4, len(unique_words))):
            for j in range(i + 1, min(5, len(unique_words))):
                keys.add(f"{c}_awpair_{unique_words[i]}_{unique_words[j]}")

    # 7. Address Number + Name Prefix
    if addr_nums and name_toks:
        t0 = name_toks[0][:3]
        for num in addr_nums[:2]:
            keys.add(f"{c}_anumt_{num}_{t0}")

    return keys

print("Extracting features for S2 and S3...")
def preprocess_df(df):
    clean_n = df["business_name"].apply(clean_text)
    clean_a = df["business_address"].apply(clean_text)
    
    name_feats = clean_n.apply(extract_name_features)
    addr_feats = clean_a.apply(extract_address_features)
    
    df["name_toks"] = [f[0] for f in name_feats]
    df["concat_str"] = [f[1] for f in name_feats]
    df["prefixes"] = [f[2] for f in name_feats]
    df["addr_nums"] = [f[0] for f in addr_feats]
    df["addr_words"] = [f[1] for f in addr_feats]
    return df

s2 = preprocess_df(s2)
s3 = preprocess_df(s3)
val_s1 = preprocess_df(val_s1)

print("Building inverted index for S2 and S3...")
start_time = time.time()
index = defaultdict(list)

for row in s2[["entity_id", "country", "name_toks", "concat_str", "prefixes", "addr_nums", "addr_words"]].itertuples(index=False):
    eid, c, n_toks, c_str, p_fx, a_nums, a_words = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
    for k in generate_blocking_keys(c, n_toks, c_str, p_fx, a_nums, a_words):
        index[k].append(eid)

for row in s3[["entity_id", "country", "name_toks", "concat_str", "prefixes", "addr_nums", "addr_words"]].itertuples(index=False):
    eid, c, n_toks, c_str, p_fx, a_nums, a_words = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
    for k in generate_blocking_keys(c, n_toks, c_str, p_fx, a_nums, a_words):
        index[k].append(eid)

print(f"Built index in {time.time() - start_time:.2f}s. Unique keys: {len(index):,}")

# Filter hyper-frequent keys (e.g. max freq 5000)
MAX_KEY_FREQ = 5000
active_index = {k: v for k, v in index.items() if len(v) <= MAX_KEY_FREQ}
print(f"Active index keys with freq <= {MAX_KEY_FREQ}: {len(active_index):,}")

print("Generating candidates for validation set...")
cand_counts = []
recalled_gt_pairs = 0

start_time = time.time()
for row in val_s1[["entity_id", "country", "name_toks", "concat_str", "prefixes", "addr_nums", "addr_words"]].itertuples(index=False):
    s1_id, c, n_toks, c_str, p_fx, a_nums, a_words = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
    
    cands = set()
    for k in generate_blocking_keys(c, n_toks, c_str, p_fx, a_nums, a_words):
        if k in active_index:
            cands.update(active_index[k])
            
    cand_counts.append(len(cands))
    
    gt_set = gt_map.get(s1_id, set())
    if gt_set:
        recalled = gt_set.intersection(cands)
        recalled_gt_pairs += len(recalled)

recall = recalled_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
avg_cands = np.mean(cand_counts)
median_cands = np.median(cand_counts)
p95_cands = np.percentile(cand_counts, 95)

print("\n" + "=" * 60)
print("HIGH RECALL BLOCKING BENCHMARK RESULTS")
print("=" * 60)
print(f"Candidate Generation Time: {time.time() - start_time:.2f}s for {len(val_s1):,} S1 entities")
print(f"GT Pair Recall: {recalled_gt_pairs:,} / {total_gt_pairs:,} ({recall * 100:.2f}%)")
print(f"Avg Candidates per S1: {avg_cands:.2f}")
print(f"Median Candidates per S1: {median_cands:.0f}")
print(f"95th Percentile Candidates per S1: {p95_cands:.0f}")
