import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
import joblib

BASE = Path("student_resource/dataset")
MODEL_PATH = Path("output/matching_model.pkl")

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

MAX_KEY_FREQ = 3000


def clean_text_fast(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_fast_keys(country, name_clean, addr_clean):
    c = str(country).lower()
    keys = []
    
    # 1. Name tokens
    raw_toks = name_clean.split()
    n_toks = []
    for t in raw_toks:
        t_stripped = re.sub(r"\.(com|net|org|in|us|uk|co|gov|edu|io|info)$", "", t)
        if len(t_stripped) >= 2 and t_stripped not in GENERIC_WORDS and not t_stripped.isdigit():
            n_toks.append(t_stripped)
            keys.append(f"{c}_tok_{t_stripped}")
            
    # 2. Concatenated name
    concat_str = "".join(n_toks)
    if len(concat_str) >= 5:
        keys.append(f"{c}_concat_{concat_str}")
        keys.append(f"{c}_p5_{concat_str[:5]}")
        
    # 3. Address features
    if addr_clean:
        nums = set(re.findall(r"\b\d+[a-z]?\b", addr_clean))
        nums = [n for n in nums if len(n) <= 6]
        
        awords = [w for w in addr_clean.split() if len(w) >= 3 and not w.isdigit() and w not in GENERIC_ADDR_WORDS]
        
        if nums and awords:
            for num in nums[:2]:
                for aw in awords[:3]:
                    keys.append(f"{c}_anw_{num}_{aw}")
                    
        distinct_aw = sorted(list(set(w for w in awords if len(w) >= 4)))
        if len(distinct_aw) >= 2:
            keys.append(f"{c}_apair_{distinct_aw[0]}_{distinct_aw[1]}")
            
    return list(set(keys))


def load_data(split):
    if split == "val":
        s1 = pd.read_csv(BASE / "val" / "val_source1.tsv", sep="\t")
        s2 = pd.read_csv(BASE / "train" / "train_source2.tsv", sep="\t")
        s3 = pd.read_csv(BASE / "train" / "train_source3.tsv", sep="\t")
    else:
        s1 = pd.read_csv(BASE / split / f"{split}_source1.tsv", sep="\t")
        s2 = pd.read_csv(BASE / split / f"{split}_source2.tsv", sep="\t")
        s3 = pd.read_csv(BASE / split / f"{split}_source3.tsv", sep="\t")

    candidates = pd.concat([s2, s3], ignore_index=True)
    return s1, candidates
def run_test():
    print("\n=== FAST FINAL TEST INFERENCE ===")

    t_start = time.time()

    s1, candidates = load_data("test")

    print(
        f"Loaded S1 ({len(s1):,}) and "
        f"Candidates ({len(candidates):,}) "
        f"in {time.time() - t_start:.2f}s"
    )

    # ---------------------------------------------------------
    # Prepare candidate lookup maps
    # ---------------------------------------------------------

    print("Building exact lookup maps...")
    t0 = time.time()

    name_address_map = defaultdict(list)
    name_map = defaultdict(list)

    for row in candidates.itertuples(index=False):

        country = "" if pd.isna(row.country) else str(row.country).lower()

        name = clean_text_fast(row.business_name)
        address = clean_text_fast(row.business_address)

        eid = str(row.entity_id)

        if name and address:
            name_address_map[
                (country, name, address)
            ].append(eid)

        if name:
            name_map[
                (country, name)
            ].append(eid)

    print(
        f"Lookup maps built in "
        f"{time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------
    # Generate TSV
    # ---------------------------------------------------------

    output_path = Path("output/matching_results.tsv")
    output_path.parent.mkdir(exist_ok=True)

    print("Generating matching_results.tsv...")

    t0 = time.time()
    matched_entities = 0

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        for i, row in enumerate(
            s1.itertuples(index=False),
            1
        ):

            s1_id = str(row.entity_id)

            country = (
                "" if pd.isna(row.country)
                else str(row.country).lower()
            )

            name = clean_text_fast(row.business_name)
            address = clean_text_fast(row.business_address)

            matches = []

            # Strongest match:
            # exact normalized name + address
            if name and address:

                matches = name_address_map.get(
                    (country, name, address),
                    []
                )

            # Fallback:
            # exact normalized name + country,
            # ONLY when unique.
            if not matches and name:

                name_matches = name_map.get(
                    (country, name),
                    []
                )

                if len(name_matches) == 1:
                    matches = name_matches

            matches = list(dict.fromkeys(matches))

            if matches:
                matched_entities += 1

            f.write(
                f"{s1_id}\t{','.join(matches)}\n"
            )

            if i % 100_000 == 0:

                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (
                    len(s1) - i
                ) / rate

                print(
                    f"Processed "
                    f"{i:,}/{len(s1):,} "
                    f"({rate:.0f} rec/s, "
                    f"~{remaining/60:.1f} min remaining)"
                )

    print("\n==============================")
    print("MATCHING RESULTS GENERATED")
    print("==============================")
    print(f"File: {output_path}")
    print(f"Rows: {len(s1):,}")
    print(
        f"Entities with matches: "
        f"{matched_entities:,}"
    )
    print(
        f"Total time: "
        f"{(time.time()-t_start)/60:.2f} minutes"
    )
    print("==============================")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/inference.py test")
        sys.exit(1)
        
    mode = sys.argv[1]
    if mode == "test":
        run_test()
    else:
        print(f"Running mode: {mode}")
        run_test()

if __name__ == "__main__":
    main()