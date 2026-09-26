import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import time


# ============================================================
# PATHS
# ============================================================

VAL_DIR = Path("student_resource/dataset/val")
TRAIN_DIR = Path("student_resource/dataset/train")


# ============================================================
# HIGH-RECALL SETTINGS
# ============================================================

MAX_TOKEN_FREQ = 20000
MAX_PREFIX_FREQ = 10000
MAX_ADDR_TOKEN_FREQ = 5000
MAX_NUMBER_FREQ = 2000
MAX_ADDR_PAIR_FREQ = 5000
MAX_NUMBER_WORD_FREQ = 5000


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation set...")

val_gt = pd.read_csv(
    VAL_DIR / "val_ground_truth.tsv",
    sep="\t"
)

val_s1 = pd.read_csv(
    VAL_DIR / "val_source1.tsv",
    sep="\t"
)

print(f"Validation S1 entities: {len(val_s1):,}")


# ============================================================
# GROUND TRUTH
# ============================================================

print("Parsing ground truth...")

gt_map = {}
total_gt_pairs = 0

for row in val_gt.itertuples(index=False):

    s1_id = row.source1_entity_id

    if pd.isna(row.matched_entity_ids):
        matches = set()
    else:
        matches = {
            x.strip()
            for x in str(row.matched_entity_ids).split(",")
            if x.strip()
        }

    gt_map[s1_id] = matches
    total_gt_pairs += len(matches)

print(
    f"Total ground truth positive pairs in val split: "
    f"{total_gt_pairs:,}"
)


# ============================================================
# LOAD SOURCE 2 / SOURCE 3
# ============================================================

print("Loading Source 2 and Source 3...")

start_time = time.time()

s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)

print(
    f"Loaded S2 ({len(s2):,}) and "
    f"S3 ({len(s3):,}) "
    f"in {time.time() - start_time:.2f}s"
)


# ============================================================
# GENERIC BUSINESS WORDS
# ============================================================

GENERIC_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited",
    "corp", "corporation", "company", "co", "plc",
    "pvt", "private", "prvt",

    "mr", "mrs", "ms", "dr",
    "shri", "smt", "sri", "shree",
    "saint", "st",

    "the", "group",

    "services", "service",
    "solutions",

    "enterprises", "enterprise",
    "business",

    "industries", "industry",
    "traders", "trading",

    "store", "stores",
    "shop", "shops",

    "center", "centre",
    "mart",

    "international", "global",

    "tech", "technologies", "technology",

    "medical", "clinic", "care",

    "holdings", "ventures",

    "systems", "system",

    "consulting", "agency",
    "management",
    "properties",

    "capital", "partners",
    "financial",

    "logistics", "express",

    "motors", "motor", "auto",

    "electronics",

    "pharma", "pharmaceuticals",

    "foods", "food",

    "textile", "textiles",

    "garments", "clothing",

    "supermarket",
    "hardware",
    "appliances",

    "agencies",

    "hospital", "hospitals",

    "trust",
    "school",
    "college",

    "foundry",

    "distributors", "distributor",

    "wholesaler", "wholesalers",

    "retail", "retails",

    "works",

    "studio", "studios",

    "labs",
    "laboratory",
    "laboratories",

    "hotel", "hotels",

    "resort", "resorts",

    "jewellers",
    "jeweller",
    "jewelers",
    "jeweler",
    "jewels",
    "jewel",

    "creations",
    "creation",

    "fashions",
    "fashion",

    "wear",
    "wears",
    "garment",

    "sports",

    "firm",

    "house",
    "home",

    "world",
    "life",

    "star",
    "royal",
    "sun",
    "moon",

    "and",
    "or",
    "of",
    "in",
    "for",
    "to"
}


# ============================================================
# GENERIC ADDRESS WORDS
# ============================================================

GENERIC_ADDR_WORDS = {
    "road",
    "street",
    "drive",
    "avenue",
    "lane",
    "boulevard",
    "highway",
    "court",

    "floor",
    "flat",
    "unit",
    "apartment",

    "no",
    "number",

    "rd",
    "st",
    "dr",
    "ave",
    "ln",
    "blvd",
    "hwy",
    "ct"
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NAME TOKEN EXTRACTION
# ============================================================

def get_name_tokens(name):

    if not name:
        return []

    tokens = []

    for token in name.split():

        # Remove common domain suffixes
        token = re.sub(
            r"\.(com|net|org|in|us|uk|co|gov|edu|io|info)$",
            "",
            token
        )

        if (
            len(token) >= 2
            and token not in GENERIC_WORDS
            and not token.isdigit()
        ):
            tokens.append(token)

    return list(dict.fromkeys(tokens))


# ============================================================
# ADDRESS TOKEN EXTRACTION
# ============================================================

def get_address_tokens(address):

    if not address:
        return []

    tokens = []

    for token in address.split():

        if (
            len(token) >= 3
            and not token.isdigit()
            and token not in GENERIC_ADDR_WORDS
        ):
            tokens.append(token)

    return list(dict.fromkeys(tokens))


# ============================================================
# ADDRESS NUMBER EXTRACTION
# ============================================================

def get_address_numbers(address):

    if not address:
        return []

    numbers = re.findall(
        r"\b\d+[a-z]?\b",
        address
    )

    return list({
        n for n in numbers
        if len(n) <= 8
    })


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(business_name, business_address):

    clean_name = clean_text(
        business_name
    )

    clean_addr = clean_text(
        business_address
    )

    # ----------------------------------------
    # Name tokens
    # ----------------------------------------

    name_tokens = get_name_tokens(
        clean_name
    )

    # ----------------------------------------
    # Compact name
    # ----------------------------------------

    compact_name = clean_name.replace(
        " ",
        ""
    )

    # ----------------------------------------
    # Concatenated meaningful tokens
    # ----------------------------------------

    concat_name = "".join(
        name_tokens
    )

    # ----------------------------------------
    # Name prefixes
    # ----------------------------------------

    prefixes = set()

    if len(concat_name) >= 4:

        prefixes.add(
            concat_name[:4]
        )

    if len(concat_name) >= 5:

        prefixes.add(
            concat_name[:5]
        )

    if len(concat_name) >= 6:

        prefixes.add(
            concat_name[:6]
        )

    for token in name_tokens:

        if len(token) >= 4:

            prefixes.add(
                token[:4]
            )

        if len(token) >= 5:

            prefixes.add(
                token[:5]
            )

    # ----------------------------------------
    # Address numbers
    # ----------------------------------------

    address_numbers = get_address_numbers(
        clean_addr
    )

    # ----------------------------------------
    # Address words
    # ----------------------------------------

    address_tokens = get_address_tokens(
        clean_addr
    )

    return {
        "clean_name": clean_name,
        "compact_name": compact_name,
        "name_tokens": name_tokens,
        "concat_name": concat_name,
        "prefixes": list(prefixes),
        "address_numbers": address_numbers,
        "address_tokens": address_tokens
    }


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(df, label):

    print(
        f"Processing {label}..."
    )

    features = []

    for row in df.itertuples(
        index=False
    ):

        features.append(
            extract_features(
                row.business_name,
                row.business_address
            )
        )

    df["clean_name"] = [
        f["clean_name"]
        for f in features
    ]

    df["compact_name"] = [
        f["compact_name"]
        for f in features
    ]

    df["name_tokens"] = [
        f["name_tokens"]
        for f in features
    ]

    df["concat_name"] = [
        f["concat_name"]
        for f in features
    ]

    df["prefixes"] = [
        f["prefixes"]
        for f in features
    ]

    df["addr_nums"] = [
        f["address_numbers"]
        for f in features
    ]

    df["addr_tokens"] = [
        f["address_tokens"]
        for f in features
    ]


print()
print("Extracting blocking features...")

feature_start = time.time()

prepare_dataframe(
    s2,
    "Source 2"
)

prepare_dataframe(
    s3,
    "Source 3"
)

prepare_dataframe(
    val_s1,
    "Validation S1"
)

print(
    f"Feature extraction completed in "
    f"{time.time() - feature_start:.2f}s"
)


# ============================================================
# INDEXES
# ============================================================

print()
print(
    "Building high-recall blocking indexes..."
)

index_start = time.time()

exact_name_index = defaultdict(list)

compact_name_index = defaultdict(list)

token_index = defaultdict(list)

token_prefix_index = defaultdict(list)

pair_index = defaultdict(list)

address_token_index = defaultdict(list)

address_number_index = defaultdict(list)

number_prefix_index = defaultdict(list)

number_word_index = defaultdict(list)

address_pair_index = defaultdict(list)


# ============================================================
# INDEX ONE RECORD
# ============================================================

def index_record(row):

    eid = row.entity_id

    country = str(
        row.country
    ).strip().lower()

    clean_name = row.clean_name

    compact_name = row.compact_name

    name_tokens = row.name_tokens

    prefixes = row.prefixes

    addr_tokens = row.addr_tokens

    addr_nums = row.addr_nums

    # ========================================================
    # 1. EXACT NORMALIZED NAME
    # ========================================================

    if clean_name:

        exact_name_index[
            f"{country}|{clean_name}"
        ].append(eid)

    # ========================================================
    # 2. COMPACT NAME
    # ========================================================

    if compact_name:

        compact_name_index[
            f"{country}|{compact_name}"
        ].append(eid)

    # ========================================================
    # 3. NAME TOKENS
    # ========================================================

    unique_tokens = list(
        set(name_tokens)
    )

    for token in unique_tokens:

        token_index[
            f"{country}|{token}"
        ].append(eid)

    # ========================================================
    # 4. NAME PREFIXES
    # ========================================================

    for prefix in set(prefixes):

        token_prefix_index[
            f"{country}|{prefix}"
        ].append(eid)

    # ========================================================
    # 5. TOKEN PAIRS
    # ========================================================

    pair_tokens = sorted(
        unique_tokens
    )[:5]

    if len(pair_tokens) >= 2:

        for i in range(
            len(pair_tokens)
        ):

            for j in range(
                i + 1,
                len(pair_tokens)
            ):

                a = pair_tokens[i]
                b = pair_tokens[j]

                pair_index[
                    f"{country}|{a}|{b}"
                ].append(eid)

    # ========================================================
    # 6. ADDRESS TOKENS
    # ========================================================

    for token in set(addr_tokens):

        address_token_index[
            f"{country}|{token}"
        ].append(eid)

    # ========================================================
    # 7. ADDRESS NUMBERS
    # ========================================================

    for number in set(addr_nums):

        address_number_index[
            f"{country}|{number}"
        ].append(eid)

    # ========================================================
    # 8. NUMBER + NAME PREFIX
    # ========================================================

    for number in addr_nums[:2]:

        for token in unique_tokens[:4]:

            prefix = token[:3]

            if prefix:

                number_prefix_index[
                    f"{country}|{number}|{prefix}"
                ].append(eid)

    # ========================================================
    # 9. NUMBER + ADDRESS WORD
    # ========================================================

    for number in addr_nums[:2]:

        for word in addr_tokens[:4]:

            number_word_index[
                f"{country}|{number}|{word}"
            ].append(eid)

    # ========================================================
    # 10. ADDRESS WORD PAIR
    # ========================================================

    address_words = sorted(
        set(
            word
            for word in addr_tokens
            if len(word) >= 4
        )
    )[:6]

    if len(address_words) >= 2:

        for i in range(
            len(address_words)
        ):

            for j in range(
                i + 1,
                len(address_words)
            ):

                a = address_words[i]
                b = address_words[j]

                address_pair_index[
                    f"{country}|{a}|{b}"
                ].append(eid)


# ============================================================
# INDEX S2
# ============================================================

print("Indexing Source 2...")

for row in s2.itertuples(
    index=False
):

    index_record(row)


# ============================================================
# INDEX S3
# ============================================================

print("Indexing Source 3...")

for row in s3.itertuples(
    index=False
):

    index_record(row)


print(
    f"Indexes built in "
    f"{time.time() - index_start:.2f}s"
)


# ============================================================
# FILTER HIGH-FREQUENCY KEYS
# ============================================================

print()
print("Filtering high-frequency keys...")


def filter_index(
    index,
    max_frequency
):

    return {
        k: v
        for k, v in index.items()
        if len(v) <= max_frequency
    }


active_token_index = filter_index(
    token_index,
    MAX_TOKEN_FREQ
)

active_prefix_index = filter_index(
    token_prefix_index,
    MAX_PREFIX_FREQ
)

active_addr_token_index = filter_index(
    address_token_index,
    MAX_ADDR_TOKEN_FREQ
)

active_number_index = filter_index(
    address_number_index,
    MAX_NUMBER_FREQ
)

active_number_word_index = filter_index(
    number_word_index,
    MAX_NUMBER_WORD_FREQ
)

active_address_pair_index = filter_index(
    address_pair_index,
    MAX_ADDR_PAIR_FREQ
)


print(
    f"Exact name keys:       "
    f"{len(exact_name_index):,}"
)

print(
    f"Compact name keys:     "
    f"{len(compact_name_index):,}"
)

print(
    f"Token keys retained:   "
    f"{len(active_token_index):,}"
)

print(
    f"Prefix keys retained:  "
    f"{len(active_prefix_index):,}"
)

print(
    f"Pair keys:             "
    f"{len(pair_index):,}"
)

print(
    f"Address token keys:    "
    f"{len(active_addr_token_index):,}"
)

print(
    f"Address number keys:   "
    f"{len(active_number_index):,}"
)

print(
    f"Number-word keys:      "
    f"{len(active_number_word_index):,}"
)

print(
    f"Address-pair keys:     "
    f"{len(active_address_pair_index):,}"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print()
print(
    "Generating high-recall candidates..."
)

candidate_start = time.time()

candidate_counts = []

recalled_gt_pairs = 0

pass_recall = {
    "exact_name": 0,
    "compact_name": 0,
    "name_token": 0,
    "token_pair": 0,
    "token_prefix": 0,
    "address_token": 0,
    "address_number": 0,
    "number_prefix": 0,
    "number_word": 0,
    "address_pair": 0
}

missed_samples = []


# ============================================================
# VALIDATION LOOP
# ============================================================

for counter, row in enumerate(
    val_s1.itertuples(index=False),
    start=1
):

    s1_id = row.entity_id

    country = str(
        row.country
    ).strip().lower()

    clean_name = row.clean_name

    compact_name = row.compact_name

    name_tokens = list(
        set(row.name_tokens)
    )

    prefixes = list(
        set(row.prefixes)
    )

    addr_tokens = list(
        set(row.addr_tokens)
    )

    addr_nums = list(
        set(row.addr_nums)
    )

    candidates = set()

    # ========================================================
    # PASS 1 — EXACT NAME
    # ========================================================

    values = exact_name_index.get(
        f"{country}|{clean_name}",
        []
    )

    candidates.update(values)

    # ========================================================
    # PASS 2 — COMPACT NAME
    # ========================================================

    values = compact_name_index.get(
        f"{country}|{compact_name}",
        []
    )

    candidates.update(values)

    # ========================================================
    # PASS 3 — NAME TOKENS
    # ========================================================

    for token in name_tokens:

        values = active_token_index.get(
            f"{country}|{token}",
            []
        )

        candidates.update(values)

    # ========================================================
    # PASS 4 — TOKEN PAIRS
    # ========================================================

    pair_tokens = sorted(
        name_tokens
    )[:5]

    if len(pair_tokens) >= 2:

        for i in range(
            len(pair_tokens)
        ):

            for j in range(
                i + 1,
                len(pair_tokens)
            ):

                a = pair_tokens[i]
                b = pair_tokens[j]

                values = pair_index.get(
                    f"{country}|{a}|{b}",
                    []
                )

                candidates.update(values)

    # ========================================================
    # PASS 5 — NAME PREFIXES
    # ========================================================

    for prefix in prefixes:

        values = active_prefix_index.get(
            f"{country}|{prefix}",
            []
        )

        candidates.update(values)

    # ========================================================
    # PASS 6 — ADDRESS TOKENS
    # ========================================================

    for token in addr_tokens:

        values = active_addr_token_index.get(
            f"{country}|{token}",
            []
        )

        candidates.update(values)

    # ========================================================
    # PASS 7 — ADDRESS NUMBERS
    # ========================================================

    for number in addr_nums:

        values = active_number_index.get(
            f"{country}|{number}",
            []
        )

        candidates.update(values)

    # ========================================================
    # PASS 8 — NUMBER + NAME PREFIX
    # ========================================================

    for number in addr_nums[:2]:

        for token in name_tokens[:4]:

            prefix = token[:3]

            if prefix:

                values = number_prefix_index.get(
                    f"{country}|{number}|{prefix}",
                    []
                )

                candidates.update(values)

    # ========================================================
    # PASS 9 — NUMBER + ADDRESS WORD
    # ========================================================

    for number in addr_nums[:2]:

        for word in addr_tokens[:4]:

            values = active_number_word_index.get(
                f"{country}|{number}|{word}",
                []
            )

            candidates.update(values)

    # ========================================================
    # PASS 10 — ADDRESS WORD PAIR
    # ========================================================

    address_words = sorted(
        set(
            word
            for word in addr_tokens
            if len(word) >= 4
        )
    )[:6]

    if len(address_words) >= 2:

        for i in range(
            len(address_words)
        ):

            for j in range(
                i + 1,
                len(address_words)
            ):

                a = address_words[i]
                b = address_words[j]

                values = active_address_pair_index.get(
                    f"{country}|{a}|{b}",
                    []
                )

                candidates.update(values)

    # ========================================================
    # CANDIDATE COUNT
    # ========================================================

    candidate_counts.append(
        len(candidates)
    )

    # ========================================================
    # GROUND TRUTH RECALL
    # ========================================================

    gt_set = gt_map.get(
        s1_id,
        set()
    )

    if gt_set:

        hits = gt_set.intersection(
            candidates
        )

        recalled_gt_pairs += len(
            hits
        )

        # ----------------------------------------------------
        # EXACT NAME DIAGNOSTIC
        # ----------------------------------------------------

        exact_candidates = set(
            exact_name_index.get(
                f"{country}|{clean_name}",
                []
            )
        )

        pass_recall["exact_name"] += len(
            gt_set.intersection(
                exact_candidates
            )
        )

        # ----------------------------------------------------
        # COMPACT NAME
        # ----------------------------------------------------

        compact_candidates = set(
            compact_name_index.get(
                f"{country}|{compact_name}",
                []
            )
        )

        pass_recall["compact_name"] += len(
            gt_set.intersection(
                compact_candidates
            )
        )

        # ----------------------------------------------------
        # NAME TOKEN
        # ----------------------------------------------------

        token_candidates = set()

        for token in name_tokens:

            token_candidates.update(
                active_token_index.get(
                    f"{country}|{token}",
                    []
                )
            )

        pass_recall["name_token"] += len(
            gt_set.intersection(
                token_candidates
            )
        )

        # ----------------------------------------------------
        # TOKEN PAIR
        # ----------------------------------------------------

        pair_candidates = set()

        if len(pair_tokens) >= 2:

            for i in range(
                len(pair_tokens)
            ):

                for j in range(
                    i + 1,
                    len(pair_tokens)
                ):

                    a = pair_tokens[i]
                    b = pair_tokens[j]

                    pair_candidates.update(
                        pair_index.get(
                            f"{country}|{a}|{b}",
                            []
                        )
                    )

        pass_recall["token_pair"] += len(
            gt_set.intersection(
                pair_candidates
            )
        )

        # ----------------------------------------------------
        # TOKEN PREFIX
        # ----------------------------------------------------

        prefix_candidates = set()

        for prefix in prefixes:

            prefix_candidates.update(
                active_prefix_index.get(
                    f"{country}|{prefix}",
                    []
                )
            )

        pass_recall["token_prefix"] += len(
            gt_set.intersection(
                prefix_candidates
            )
        )

        # ----------------------------------------------------
        # ADDRESS TOKEN
        # ----------------------------------------------------

        address_candidates = set()

        for token in addr_tokens:

            address_candidates.update(
                active_addr_token_index.get(
                    f"{country}|{token}",
                    []
                )
            )

        pass_recall["address_token"] += len(
            gt_set.intersection(
                address_candidates
            )
        )

        # ----------------------------------------------------
        # ADDRESS NUMBER
        # ----------------------------------------------------

        number_candidates = set()

        for number in addr_nums:

            number_candidates.update(
                active_number_index.get(
                    f"{country}|{number}",
                    []
                )
            )

        pass_recall["address_number"] += len(
            gt_set.intersection(
                number_candidates
            )
        )

        # ----------------------------------------------------
        # NUMBER + NAME PREFIX
        # ----------------------------------------------------

        number_prefix_candidates = set()

        for number in addr_nums[:2]:

            for token in name_tokens[:4]:

                prefix = token[:3]

                if prefix:

                    number_prefix_candidates.update(
                        number_prefix_index.get(
                            f"{country}|{number}|{prefix}",
                            []
                        )
                    )

        pass_recall["number_prefix"] += len(
            gt_set.intersection(
                number_prefix_candidates
            )
        )

        # ----------------------------------------------------
        # NUMBER + ADDRESS WORD
        # ----------------------------------------------------

        number_word_candidates = set()

        for number in addr_nums[:2]:

            for word in addr_tokens[:4]:

                number_word_candidates.update(
                    active_number_word_index.get(
                        f"{country}|{number}|{word}",
                        []
                    )
                )

        pass_recall["number_word"] += len(
            gt_set.intersection(
                number_word_candidates
            )
        )

        # ----------------------------------------------------
        # ADDRESS PAIR
        # ----------------------------------------------------

        address_pair_candidates = set()

        if len(address_words) >= 2:

            for i in range(
                len(address_words)
            ):

                for j in range(
                    i + 1,
                    len(address_words)
                ):

                    a = address_words[i]
                    b = address_words[j]

                    address_pair_candidates.update(
                        active_address_pair_index.get(
                            f"{country}|{a}|{b}",
                            []
                        )
                    )

        pass_recall["address_pair"] += len(
            gt_set.intersection(
                address_pair_candidates
            )
        )

        # ----------------------------------------------------
        # MISSED EXAMPLES
        # ----------------------------------------------------

        missed = gt_set.difference(
            candidates
        )

        if (
            missed
            and len(missed_samples) < 20
        ):

            for missed_id in missed:

                missed_samples.append(
                    (
                        s1_id,
                        missed_id
                    )
                )

    # ========================================================
    # PROGRESS
    # ========================================================

    if counter % 5000 == 0:

        elapsed = (
            time.time()
            - candidate_start
        )

        print(
            f"  Processed "
            f"{counter:,}/"
            f"{len(val_s1):,} "
            f"| {elapsed:.1f}s"
        )


# ============================================================
# FINAL METRICS
# ============================================================

candidate_time = (
    time.time()
    - candidate_start
)

recall = (
    recalled_gt_pairs
    / total_gt_pairs
    if total_gt_pairs
    else 0
)

avg_candidates = np.mean(
    candidate_counts
)

median_candidates = np.median(
    candidate_counts
)

p95_candidates = np.percentile(
    candidate_counts,
    95
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("BLOCKING BENCHMARK RESULTS")
print("=" * 70)

print(
    f"Candidate Generation Time: "
    f"{candidate_time:.2f}s"
)

print(
    f"GT Pair Recall: "
    f"{recalled_gt_pairs:,} / "
    f"{total_gt_pairs:,} "
    f"({recall * 100:.2f}%)"
)

print(
    f"Avg Candidates per S1: "
    f"{avg_candidates:.2f}"
)

print(
    f"Median Candidates per S1: "
    f"{median_candidates:.0f}"
)

print(
    f"95th Percentile Candidates per S1: "
    f"{p95_candidates:.0f}"
)


# ============================================================
# PASS-WISE RECALL
# ============================================================

print()
print("=" * 70)
print("PASS-WISE GROUND TRUTH RECALL")
print("=" * 70)

for name, count in pass_recall.items():

    percentage = (
        count / total_gt_pairs * 100
        if total_gt_pairs
        else 0
    )

    print(
        f"{name:<20} "
        f"{count:>10,} "
        f"({percentage:>7.2f}%)"
    )


# ============================================================
# MISSED PAIRS
# ============================================================

print()
print("=" * 70)
print("MISSED GROUND TRUTH PAIRS")
print("=" * 70)

if missed_samples:

    for s1_id, missed_id in missed_samples:

        print(
            f"S1: {s1_id} "
            f"--> Missing: {missed_id}"
        )

else:

    print(
        "No missed ground-truth pairs!"
    )


# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CANDIDATE DISTRIBUTION")
print("=" * 70)

print(
    f"Minimum candidates: "
    f"{np.min(candidate_counts):.0f}"
)

print(
    f"Maximum candidates: "
    f"{np.max(candidate_counts):.0f}"
)

print(
    f"Mean candidates: "
    f"{np.mean(candidate_counts):.2f}"
)

print(
    f"Median candidates: "
    f"{np.median(candidate_counts):.0f}"
)

print(
    f"P90 candidates: "
    f"{np.percentile(candidate_counts, 90):.0f}"
)

print(
    f"P95 candidates: "
    f"{np.percentile(candidate_counts, 95):.0f}"
)

print(
    f"P99 candidates: "
    f"{np.percentile(candidate_counts, 99):.0f}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)