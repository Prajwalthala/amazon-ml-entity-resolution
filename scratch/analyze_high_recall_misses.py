import pandas as pd
import re
from pathlib import Path
from collections import defaultdict, Counter

VAL_DIR = Path("student_resource/dataset/val")
TRAIN_DIR = Path("student_resource/dataset/train")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading validation set...")
val_gt = pd.read_csv(
    VAL_DIR / "val_ground_truth.tsv",
    sep="\t"
)
val_s1 = pd.read_csv(
    VAL_DIR / "val_source1.tsv",
    sep="\t"
)

print("Loading S2 and S3...")
s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)
s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)

# ---------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------

gt_map = {}

for _, row in val_gt.iterrows():
    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        gt_map[s1_id] = set()
    else:
        gt_map[s1_id] = {
            x.strip()
            for x in str(row["matched_entity_ids"]).split(",")
            if x.strip()
        }

total_gt_pairs = sum(len(v) for v in gt_map.values())

print(f"Total GT pairs: {total_gt_pairs:,}")

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

GENERIC_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited",
    "corp", "corporation", "company", "co", "plc",
    "pvt", "private", "prvt",

    "mr", "mrs", "ms", "dr", "shri", "smt",
    "sri", "shree", "saint", "st",

    "the", "group", "services", "service",
    "solutions", "enterprises", "enterprise",
    "business", "industries", "industry",
    "traders", "trading", "store", "stores",
    "shop", "shops", "center", "centre",
    "mart", "international", "global",

    "tech", "technologies", "technology",
    "medical", "clinic", "care",
    "holdings", "ventures", "systems", "system",
    "consulting", "agency", "management",
    "properties", "capital", "partners",
    "financial", "logistics", "express",
    "motors", "auto", "electronics",
    "pharma", "pharmaceuticals",
    "foods", "food", "textile", "textiles",
    "garments", "clothing", "supermarket",
    "hardware", "appliances", "agencies",
    "hospital", "hospitals", "trust",
    "school", "college", "foundry",
    "distributors", "distributor",
    "wholesaler", "wholesalers",
    "retail", "retails", "works",
    "studio", "studios", "labs",
    "laboratory", "laboratories",
    "hotel", "hotels", "resort", "resorts",
    "jewellers", "jeweller", "jewelers",
    "jeweler", "jewels", "jewel",
    "creations", "creation", "fashions",
    "fashion", "wear", "wears",
    "garment", "sports", "firm",
    "house", "home", "world", "life",
    "star", "royal", "sun", "moon",
    "and", "or", "of", "in", "for", "to"
}

GENERIC_ADDR_WORDS = {
    "road", "street", "drive", "avenue",
    "lane", "boulevard", "highway",
    "court", "floor", "flat", "unit",
    "apartment", "no", "number",
    "rd", "st", "dr", "ave", "ln",
    "blvd", "hwy", "ct"
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_name_features(name_clean):

    if not name_clean:
        return [], "", []

    raw_tokens = name_clean.split()

    tokens = []

    for t in raw_tokens:

        t_clean = re.sub(
            r"\.(com|net|org|in|us|uk|co|gov|edu|io|info)$",
            "",
            t
        )

        if (
            len(t_clean) >= 2
            and t_clean not in GENERIC_WORDS
            and not t_clean.isdigit()
        ):
            tokens.append(t_clean)

    concat_str = "".join(tokens)

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

    nums = set(
        re.findall(r"\b\d+[a-z]?\b", addr_clean)
    )

    nums = [
        n for n in nums
        if len(n) <= 6
    ]

    words = [
        w
        for w in addr_clean.split()
        if (
            len(w) >= 3
            and not w.isdigit()
            and w not in GENERIC_ADDR_WORDS
        )
    ]

    return nums, words[:6]


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def preprocess_df(df):

    result = []

    for _, row in df.iterrows():

        name_clean = clean_text(row["business_name"])
        addr_clean = clean_text(row["business_address"])

        name_toks, concat_str, prefixes = \
            extract_name_features(name_clean)

        addr_nums, addr_words = \
            extract_address_features(addr_clean)

        result.append({
            "entity_id": row["entity_id"],
            "country": str(row["country"]).lower(),
            "name_toks": name_toks,
            "concat_str": concat_str,
            "prefixes": prefixes,
            "addr_nums": addr_nums,
            "addr_words": addr_words,
            "business_name": row["business_name"],
            "business_address": row["business_address"]
        })

    return result


print("Extracting features...")

s2_records = preprocess_df(s2)
s3_records = preprocess_df(s3)
s1_records = preprocess_df(val_s1)

# ---------------------------------------------------------
# EXACT SAME BLOCKING LOGIC
# ---------------------------------------------------------

def generate_blocking_keys(
    country,
    name_toks,
    concat_str,
    prefixes,
    addr_nums,
    addr_words
):

    keys = set()

    c = country.lower()

    # 1. Distinctive name tokens
    for t in set(name_toks):
        keys.add(f"{c}_tok_{t}")

    # 2. Token pairs
    if len(name_toks) >= 2:

        sorted_t = sorted(set(name_toks))

        for i in range(min(4, len(sorted_t))):
            for j in range(
                i + 1,
                min(5, len(sorted_t))
            ):
                keys.add(
                    f"{c}_pair_{sorted_t[i]}_{sorted_t[j]}"
                )

    # 3. Concatenated name
    if len(concat_str) >= 5:
        keys.add(
            f"{c}_concat_{concat_str}"
        )

    # 4. Prefixes
    for p in set(prefixes):
        keys.add(
            f"{c}_p5_{p}"
        )

    # 5. Address number + address word
    if addr_nums and addr_words:

        for num in addr_nums[:2]:

            for aword in addr_words[:3]:

                keys.add(
                    f"{c}_anw_{num}_{aword}"
                )

    # 6. Address number + name prefix
    if addr_nums and name_toks:

        t0 = name_toks[0][:3]

        for num in addr_nums[:2]:

            keys.add(
                f"{c}_anumt_{num}_{t0}"
            )

    return keys


# ---------------------------------------------------------
# BUILD INDEX
# ---------------------------------------------------------

print("Building inverted index...")

index = defaultdict(list)

for rec in s2_records + s3_records:

    keys = generate_blocking_keys(
        rec["country"],
        rec["name_toks"],
        rec["concat_str"],
        rec["prefixes"],
        rec["addr_nums"],
        rec["addr_words"]
    )

    for key in keys:
        index[key].append(rec["entity_id"])


MAX_KEY_FREQ = 5000

active_index = {
    k: v
    for k, v in index.items()
    if len(v) <= MAX_KEY_FREQ
}

print(
    f"Active index keys: {len(active_index):,}"
)

# ---------------------------------------------------------
# ENTITY LOOKUP
# ---------------------------------------------------------

entity_map = {}

for rec in s2_records:
    entity_map[rec["entity_id"]] = rec

for rec in s3_records:
    entity_map[rec["entity_id"]] = rec


# ---------------------------------------------------------
# ANALYZE MISSED GT PAIRS
# ---------------------------------------------------------

print("\nFinding missed GT pairs...")

missed = []

for s1 in s1_records:

    s1_id = s1["entity_id"]

    keys = generate_blocking_keys(
        s1["country"],
        s1["name_toks"],
        s1["concat_str"],
        s1["prefixes"],
        s1["addr_nums"],
        s1["addr_words"]
    )

    candidates = set()

    for key in keys:

        if key in active_index:
            candidates.update(
                active_index[key]
            )

    gt_set = gt_map.get(
        s1_id,
        set()
    )

    missing = gt_set - candidates

    for mid in missing:

        missed.append(
            (s1, mid)
        )


print(
    f"\nMissed GT pairs: {len(missed):,}"
)

print(
    f"Recall: "
    f"{(1 - len(missed) / total_gt_pairs) * 100:.2f}%"
)


# ---------------------------------------------------------
# PATTERN ANALYSIS
# ---------------------------------------------------------

categories = Counter()


for s1, mid in missed:

    m = entity_map.get(mid)

    if not m:
        categories["missing_entity"] += 1
        continue

    s1_tokens = set(s1["name_toks"])
    m_tokens = set(m["name_toks"])

    # Common token
    if s1_tokens & m_tokens:
        categories["common_name_token"] += 1
        continue

    # Substring
    substring = False

    for a in s1_tokens:

        for b in m_tokens:

            if (
                len(a) >= 4
                and len(b) >= 4
                and (a in b or b in a)
            ):
                substring = True
                break

        if substring:
            break

    if substring:
        categories["name_substring"] += 1
        continue

    # Address
    common_nums = (
        set(s1["addr_nums"])
        &
        set(m["addr_nums"])
    )

    common_words = (
        set(s1["addr_words"])
        &
        set(m["addr_words"])
    )

    if common_nums and common_words:
        categories[
            "address_number_and_word"
        ] += 1
        continue

    if common_words:
        categories[
            "address_word_only"
        ] += 1
        continue

    if common_nums:
        categories[
            "address_number_only"
        ] += 1
        continue

    # Different scripts / no shared tokens
    categories[
        "other_difficult"
    ] += 1


print("\n" + "=" * 60)
print("MISSED PAIR BREAKDOWN")
print("=" * 60)

for category, count in categories.most_common():

    print(
        f"{category:30s}: "
        f"{count:6,} "
        f"({count / len(missed) * 100:.2f}%)"
    )


# ---------------------------------------------------------
# SHOW EXAMPLES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE MISSED PAIRS")
print("=" * 60)

for i, (s1, mid) in enumerate(missed[:30], 1):

    m = entity_map.get(mid)

    if not m:
        continue

    print(f"\n[{i}]")
    print(f"S1 ID:   {s1['entity_id']}")
    print(f"S1 Name: {s1['business_name']}")
    print(f"S1 Addr: {s1['business_address']}")

    print(f"M ID:    {mid}")
    print(f"M Name:  {m['business_name']}")
    print(f"M Addr:  {m['business_address']}")

    print(
        f"S1 name tokens: {s1['name_toks']}"
    )
    print(
        f"M name tokens:  {m['name_toks']}"
    )

    print(
        f"S1 addr nums: {s1['addr_nums']}"
    )
    print(
        f"M addr nums:  {m['addr_nums']}"
    )

    print(
        f"S1 addr words: {s1['addr_words']}"
    )
    print(
        f"M addr words:  {m['addr_words']}"
    )
