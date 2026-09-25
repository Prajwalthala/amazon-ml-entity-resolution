import pandas as pd
from pathlib import Path

TRAIN_DIR = Path("student_resource/dataset/train")

print("Loading data...")
s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t").set_index("entity_id")
s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t").set_index("entity_id")
s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t").set_index("entity_id")

gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")
gt_matched = gt[gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "")].sample(15, random_state=42)

for idx, row in gt_matched.iterrows():
    s1_id = row["source1_entity_id"]
    matched_ids = [i.strip() for i in row["matched_entity_ids"].split(",")]
    
    s1_rec = s1.loc[s1_id]
    print("=" * 80)
    print(f"S1 ID: {s1_id} | Country: {s1_rec['country']}")
    print(f"  Name   : {s1_rec['business_name']}")
    print(f"  Address: {s1_rec['business_address']}")
    print("-" * 40)
    
    for mid in matched_ids:
        if mid.startswith("S2-"):
            rec = s2.loc[mid]
        else:
            rec = s3.loc[mid]
        print(f"  {mid} ({rec['country']}):")
        print(f"    Name   : {rec['business_name']}")
        print(f"    Address: {rec['business_address']}")
    print()
