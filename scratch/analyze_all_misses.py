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

print("Loading S2 and S3...")
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")

# Parse GT
gt_map = {}
total_gt_pairs = 0
for _, row in val_gt.iterrows():
    s1_id = row["source1_entity_id"]
    m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
    m_set = set(i.strip() for i in m_str.split(",") if i.strip())
    gt_map[s1_id] = m_set
    total_gt_pairs += len(m_set)

print(f"Total GT pairs: {total_gt_pairs:,}")

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
    return [t for t in tokens if len(t) >= 2 and t not in GENERIC_WORDS and not t.isdigit()]

def extract_address_numbers(addr_clean):
    if not addr_clean:
        return []
    nums = re.findall(r"\b\d+[a-z]?\b", addr_clean)
    return [n for n in nums if len(n) <= 6]

def extract_address_words(addr_clean):
    if not addr_clean:
        return []
    words = [w for w in addr_clean.split() if len(w) >= 3 and not w.isdigit() and w not in {"road", "street", "drive", "avenue", "lane", "boulevard", "highway", "court", "floor", "flat", "unit", "apartment", "no", "number"}]
    return words

# Store entity lookups
s2_map = s2.set_index("entity_id").to_dict(orient="index")
s3_map = s3.set_index("entity_id").to_dict(orient="index")

def get_entity(eid):
    if eid in s2_map:
        return s2_map[eid]
    return s3_map.get(eid)

# Preprocess val S1
val_s1_records = val_s1.to_dict(orient="records")

categories = defaultdict(int)

print("Analyzing sample of GT pairs...")
sample_gt = []
for row in val_s1_records:
    s1_id = row["entity_id"]
    m_set = gt_map.get(s1_id, set())
    for mid in m_set:
        sample_gt.append((row, mid))

print(f"Total GT pairs to analyze: {len(sample_gt):,}")

for s1_rec, mid in sample_gt[:10000]:
    m_rec = get_entity(mid)
    if not m_rec:
        categories["missing_entity_data"] += 1
        continue
    
    s1_c = str(s1_rec["country"]).lower()
    m_c = str(m_rec["country"]).lower()
    
    if s1_c != m_c:
        categories["country_mismatch"] += 1
        
    s1_n = clean_text(s1_rec["business_name"])
    m_n = clean_text(m_rec["business_name"])
    
    s1_toks = extract_distinctive_tokens(s1_n)
    m_toks = extract_distinctive_tokens(m_n)
    
    common_toks = set(s1_toks).intersection(set(m_toks))
    if common_toks:
        categories["has_common_token"] += 1
        continue
        
    # Check if domain/handle cleaning or concatenation allows token match
    s1_raw_toks = re.sub(r"[^\w]", " ", str(s1_rec["business_name"]).lower()).split()
    m_raw_toks = re.sub(r"[^\w]", " ", str(m_rec["business_name"]).lower()).split()
    
    # Strip com, net, org, www
    s1_strip = [t for t in s1_toks if t not in {"com", "net", "org", "www"}]
    m_strip = [t for t in m_toks if t not in {"com", "net", "org", "www"}]
    
    # Substring / Prefix match in long words
    has_sub = False
    for t1 in s1_strip:
        for t2 in m_strip:
            if len(t1) >= 4 and (t1 in t2 or t2 in t1):
                has_sub = True
                break
        if has_sub:
            break
            
    if has_sub:
        categories["has_substring_match"] += 1
        continue
        
    # Check address numbers + words
    s1_a = clean_text(s1_rec["business_address"])
    m_a = clean_text(m_rec["business_address"])
    
    s1_nums = set(extract_address_numbers(s1_a))
    m_nums = set(extract_address_numbers(m_a))
    common_nums = s1_nums.intersection(m_nums)
    
    s1_awords = set(extract_address_words(s1_a))
    m_awords = set(extract_address_words(m_a))
    common_awords = s1_awords.intersection(m_awords)
    
    if common_nums and common_awords:
        categories["addr_num_and_word_match"] += 1
        continue
    elif common_nums:
        categories["addr_num_only_match"] += 1
        continue
    elif common_awords:
        categories["addr_word_only_match"] += 1
        continue
        
    # Prefix 4-gram match on names
    s1_p4 = set(t[:4] for t in s1_strip if len(t) >= 4)
    m_p4 = set(t[:4] for t in m_strip if len(t) >= 4)
    if s1_p4.intersection(m_p4):
        categories["name_4gram_prefix_match"] += 1
        continue
        
    categories["other_difficult"] += 1

print("\nGT Pair Match Breakdown (10,000 sample):")
for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:30s}: {v:6d} ({v/10000*100:.2f}%)")
