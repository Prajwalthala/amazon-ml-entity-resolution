import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re

from blocking import (
    generate_blocking_keys,
    extract_name_features,
    extract_address_features,
    clean_text,
)

BASE = Path("student_resource/dataset")

def normalize_for_matching(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading test data...")
t0 = time.time()
s1 = pd.read_csv(BASE / "test" / "test_source1.tsv", sep="\t")
s2 = pd.read_csv(BASE / "test" / "test_source2.tsv", sep="\t")
s3 = pd.read_csv(BASE / "test" / "test_source3.tsv", sep="\t")
print(f"Loaded S1 ({len(s1):,}), S2 ({len(s2):,}), S3 ({len(s3):,}) in {time.time()-t0:.2f}s")

candidates = pd.concat([s2, s3], ignore_index=True)
del s2, s3  # Free memory

print("Pre-processing candidate arrays...")
t0 = time.time()
cand_eids = candidates["entity_id"].values.astype(str)
cand_ctrys = candidates["country"].fillna("").astype(str).str.lower().values
cand_names = candidates["business_name"].fillna("").astype(str).values
cand_addrs = candidates["business_address"].fillna("").astype(str).values

# Vectorized / fast list extraction for candidates
print("Extracting candidate features...")
cand_norm_names = [normalize_for_matching(n) for n in cand_names]
cand_norm_addrs = [normalize_for_matching(a) for a in cand_addrs]

# Build inverted index and exact maps
print("Building fast inverted index & exact lookup maps...")
index = defaultdict(list)
exact_name_addr_map = defaultdict(list)
exact_name_map = defaultdict(list)

MAX_KEY_FREQ = 3000

for i in range(len(candidates)):
    c = cand_ctrys[i]
    n_norm = cand_norm_names[i]
    a_norm = cand_norm_addrs[i]
    eid = cand_eids[i]
    
    if n_norm and a_norm:
        exact_name_addr_map[(c, n_norm, a_norm)].append(eid)
    if n_norm:
        exact_name_map[(c, n_norm)].append(eid)
        
    # Blocking features
    n_toks, c_str, p_fx = extract_name_features(cand_names[i])
    a_nums, a_words = extract_address_features(cand_addrs[i])
    keys = generate_blocking_keys(c, n_toks, c_str, p_fx, a_nums, a_words)
    
    for k in keys:
        if len(index[k]) < MAX_KEY_FREQ:
            index[k].append(eid)

print(f"Indexed {len(candidates):,} candidates in {time.time()-t0:.2f}s. Total keys: {len(index):,}")

# Filter active index keys
active_index = {k: v for k, v in index.items() if len(v) <= MAX_KEY_FREQ}
print(f"Active index keys (freq <= {MAX_KEY_FREQ}): {len(active_index):,}")

print("\nRunning fast candidate matching benchmark on first 50,000 S1 records...")
s1_subset = s1.iloc[:50000]

s1_eids = s1_subset["entity_id"].values.astype(str)
s1_ctrys = s1_subset["country"].fillna("").astype(str).str.lower().values
s1_names = s1_subset["business_name"].fillna("").astype(str).values
s1_addrs = s1_subset["business_address"].fillna("").astype(str).values

t0 = time.time()
match_counts = []

for i in range(len(s1_subset)):
    eid = s1_eids[i]
    c = s1_ctrys[i]
    name = s1_names[i]
    addr = s1_addrs[i]
    
    n_norm = normalize_for_matching(name)
    a_norm = normalize_for_matching(addr)
    
    matched_ids = set()
    
    # 1. Exact Name + Address Match
    if n_norm and a_norm and (c, n_norm, a_norm) in exact_name_addr_map:
        matched_ids.update(exact_name_addr_map[(c, n_norm, a_norm)])
        
    # 2. Exact Name Match
    if n_norm and (c, n_norm) in exact_name_map:
        matched_ids.update(exact_name_map[(c, n_norm)])
        
    # 3. Multi-key agreement match if no exact match found or to supplement
    if len(matched_ids) < 5:
        n_toks, c_str, p_fx = extract_name_features(name)
        a_nums, a_words = extract_address_features(addr)
        keys = generate_blocking_keys(c, n_toks, c_str, p_fx, a_nums, a_words)
        
        key_counts = Counter()
        for k in keys:
            if k in active_index:
                for cand_eid in active_index[k]:
                    key_counts[cand_eid] += 1
                    
        # Add candidates with >= 2 matching blocking keys
        for cand_eid, count in key_counts.items():
            if count >= 2:
                matched_ids.add(cand_eid)
                if len(matched_ids) >= 10:
                    break

    match_counts.append(len(matched_ids))

elapsed = time.time() - t0
rate = len(s1_subset) / elapsed
est_total_sec = len(s1) / rate

print(f"\nCompleted 50,000 S1 records in {elapsed:.2f}s ({rate:.1f} records/sec)")
print(f"Estimated time for all 1.73 MILLION records: {est_total_sec:.1f}s ({est_total_sec/60:.2f} minutes!)")
print(f"Avg matches per S1: {np.mean(match_counts):.2f}, Non-zero matches: {np.sum(np.array(match_counts)>0)/len(match_counts)*100:.2f}%")
