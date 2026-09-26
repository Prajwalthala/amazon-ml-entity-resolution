import re


# ============================================================
# Generic business/name words
# ============================================================

GENERIC_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited",
    "corp", "corporation", "company", "co", "plc",
    "pvt", "private", "prvt",
    "mr", "mrs", "ms", "dr", "shri", "smt", "sri", "shree", "saint", "st",
    "the", "group", "services", "service", "solutions",
    "enterprises", "enterprise", "business",
    "industries", "industry", "traders", "trading",
    "store", "stores", "shop", "shops", "center", "centre",
    "mart", "international", "global",
    "tech", "technologies", "technology",
    "medical", "clinic", "care",
    "holdings", "ventures",
    "systems", "system",
    "consulting", "agency", "management",
    "properties", "capital", "partners",
    "financial", "logistics", "express",
    "motors", "auto",
    "electronics", "pharma", "pharmaceuticals",
    "foods", "food",
    "textile", "textiles", "garments", "clothing",
    "supermarket", "hardware", "appliances",
    "agencies", "hospital", "hospitals",
    "trust", "school", "college",
    "foundry", "distributors", "distributor",
    "wholesaler", "wholesalers",
    "retail", "retails",
    "works", "studio", "studios",
    "labs", "laboratory", "laboratories",
    "hotel", "hotels",
    "resort", "resorts",
    "jewellers", "jeweller", "jewelers", "jeweler",
    "jewels", "jewel",
    "creations", "creation",
    "fashions", "fashion",
    "wear", "wears", "garment",
    "sports", "firm",
    "house", "home", "world",
    "life", "star", "royal",
    "sun", "moon",
    "and", "or", "of", "in", "for", "to"
}


# ============================================================
# Generic address words
# ============================================================

GENERIC_ADDR_WORDS = {
    "road", "street", "drive", "avenue", "lane",
    "boulevard", "highway", "court",
    "floor", "flat", "unit", "apartment",
    "no", "number",
    "rd", "st", "dr", "ave", "ln",
    "blvd", "hwy", "ct"
}


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):
    """
    Normalize text for blocking.

    Keeps Unicode letters/numbers so that names in
    Hindi, Telugu, Tamil, etc. are not destroyed.
    """

    if text is None:
        return ""

    text = str(text).lower()

    cleaned = []

    for char in text:
        if char.isalnum() or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Name features
# ============================================================

def extract_name_features(name_clean):
    """
    Returns:

        name_tokens
        concatenated_name
        prefixes
    """

    if not name_clean:
        return [], "", []

    raw_tokens = name_clean.split()

    tokens = []

    for token in raw_tokens:

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

    concat_str = "".join(tokens)

    prefixes = []

    if len(concat_str) >= 5:
        prefixes.append(concat_str[:5])
        prefixes.append(concat_str[:6])

    for token in tokens:

        if len(token) >= 6:
            prefixes.append(token[:5])

    return tokens, concat_str, list(set(prefixes))


# ============================================================
# Address features
# ============================================================

def extract_address_features(address_clean):
    """
    Extract useful address numbers and words.
    """

    if not address_clean:
        return [], []

    nums = set(
        re.findall(
            r"\b\d+[a-z]?\b",
            address_clean
        )
    )

    nums = [
        n for n in nums
        if len(n) <= 6
    ]

    words = [
        word
        for word in address_clean.split()
        if (
            len(word) >= 3
            and not word.isdigit()
            and word not in GENERIC_ADDR_WORDS
        )
    ]

    return nums, words[:6]


# ============================================================
# Blocking key generation
# ============================================================

def generate_blocking_keys(
    country,
    name_tokens,
    concat_str,
    prefixes,
    address_numbers,
    address_words
):
    """
    Generate multiple blocking keys for one entity.

    Multiple keys improve recall while keeping the candidate
    generation stage manageable.
    """

    keys = set()

    country = str(country).lower()

    # --------------------------------------------------------
    # 1. Distinctive name token
    # --------------------------------------------------------

    for token in set(name_tokens):

        keys.add(
            f"{country}_tok_{token}"
        )


    # --------------------------------------------------------
    # 2. Name token pairs
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        unique_tokens = sorted(
            set(name_tokens)
        )

        for i in range(
            min(4, len(unique_tokens))
        ):

            for j in range(
                i + 1,
                min(5, len(unique_tokens))
            ):

                keys.add(
                    f"{country}_pair_"
                    f"{unique_tokens[i]}_"
                    f"{unique_tokens[j]}"
                )


    # --------------------------------------------------------
    # 3. Concatenated name
    # --------------------------------------------------------

    if len(concat_str) >= 5:

        keys.add(
            f"{country}_concat_{concat_str}"
        )


    # --------------------------------------------------------
    # 4. Name prefixes
    # --------------------------------------------------------

    for prefix in set(prefixes):

        keys.add(
            f"{country}_p5_{prefix}"
        )


    # --------------------------------------------------------
    # 5. Address number + address word
    # --------------------------------------------------------

    if address_numbers and address_words:

        for number in address_numbers[:2]:

            for word in address_words[:3]:

                keys.add(
                    f"{country}_anw_"
                    f"{number}_{word}"
                )


    # --------------------------------------------------------
    # 6. Address word pairs
    # --------------------------------------------------------

    if len(address_words) >= 2:

        unique_words = list(
            dict.fromkeys(address_words)
        )

        for i in range(
            min(4, len(unique_words))
        ):

            for j in range(
                i + 1,
                min(5, len(unique_words))
            ):

                keys.add(
                    f"{country}_awpair_"
                    f"{unique_words[i]}_"
                    f"{unique_words[j]}"
                )


    # --------------------------------------------------------
    # 7. Address number + name prefix
    # --------------------------------------------------------

    if address_numbers and name_tokens:

        name_prefix = name_tokens[0][:3]

        for number in address_numbers[:2]:

            keys.add(
                f"{country}_anumt_"
                f"{number}_{name_prefix}"
            )


    return keys
