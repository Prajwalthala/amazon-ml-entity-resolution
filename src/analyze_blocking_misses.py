import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

VAL_DIR = Path("student_resource/dataset/val")
TRAIN_DIR = Path("student_resource/dataset/train")

val_gt = pd.read_csv(VAL_DIR / "val_ground_truth.tsv", sep="\t")
val_s1 = pd.read_csv(VAL_DIR / "val_source1.tsv", sep="\t").set_index("entity_id")
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t").set_index("entity_id")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t").set_index("entity_id")

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

def get_blocking_keys(c, d_toks):
    keys = []
    for t in set(d_toks):
        keys.append(f"{c}_tok_{t}")
    return set(keys)

print("Analyzing misses...")
miss_count = 0
for idx, row in val_gt.iterrows():
    s1_id = row["source1_entity_id"]
    if pd.isna(row["matched_entity_ids"]) or not str(row["matched_entity_ids"]).strip():
        continue
    
    s1_rec = val_s1.loc[s1_id]
    c = str(s1_rec["country"]).lower()
    s1_name_clean = clean_text(s1_rec["business_name"])
    s1_tokens = extract_distinctive_tokens(s1_name_clean)
    
    matched_ids = [i.strip() for i in row["matched_entity_ids"].split(",")]
    
    for mid in matched_ids:
        rec = s2.loc[mid] if mid.startswith("S2-") else s3.loc[mid]
        m_name_clean = clean_text(rec["business_name"])
        m_tokens = extract_distinctive_tokens(m_name_clean)
        
        common_toks = set(s1_tokens).intersection(m_tokens)
        if not common_toks:
            miss_count += 1
            if miss_count <= 25:
                print("=" * 80)
                print(f"MISS #{miss_count} | S1 ID: {s1_id} -> Matched ID: {mid}")
                print(f"  S1 Name       : {s1_rec['business_name']}")
                print(f"  S1 Clean Toks : {s1_tokens}")
                print(f"  S1 Address    : {s1_rec['business_address']}")
                print(f"  M Name        : {rec['business_name']}")
                print(f"  M Clean Toks  : {m_tokens}")
                print(f"  M Address     : {rec['business_address']}")
