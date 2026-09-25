import re
import pandas as pd


GENERIC_WORDS = {
    # Legal/business terms
    "llc",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",

    # India-specific
    "pvt",
    "private",
    "prvt",

    # Titles
    "mr",
    "mrs",
    "ms",
    "dr",
    "shri",
    "smt",

    # Generic business words
    "the",
    "group",
    "services",
    "service",
    "solutions",
    "enterprises",
    "enterprise",
    "business",
    "industries",
}


def get_name_tokens(name):
    """Get meaningful tokens from normalized business name."""

    if not name:
        return []

    tokens = name.split()

    return [
        token
        for token in tokens
        if token not in GENERIC_WORDS and len(token) >= 2
    ]


def get_address_tokens(address):
    """Get useful tokens from normalized address."""

    if not address:
        return []

    return [
        token
        for token in address.split()
        if len(token) >= 2
    ]


def create_blocking_keys(row):
    """
    Create multiple blocking keys using:
    - country
    - business name
    - address
    """

    country = str(row["country"]).strip().lower()

    name_tokens = get_name_tokens(
        row["name_normalized"]
    )

    address_tokens = get_address_tokens(
        row["address_normalized"]
    )

    keys = set()

    # -------------------------
    # NAME BLOCKING
    # -------------------------

    for token in name_tokens[:3]:
        keys.add(f"{country}_name_{token}")

    # First two name tokens
    if len(name_tokens) >= 2:
        keys.add(
            f"{country}_name_{name_tokens[0]}_{name_tokens[1]}"
        )

    # -------------------------
    # ADDRESS BLOCKING
    # -------------------------

    # Use address tokens
    for token in address_tokens[:3]:
        keys.add(f"{country}_addr_{token}")

    # -------------------------
    # NAME + ADDRESS
    # -------------------------

    if name_tokens and address_tokens:
        keys.add(
            f"{country}_nameaddr_"
            f"{name_tokens[0]}_"
            f"{address_tokens[0]}"
        )

    return list(keys)


def add_blocking_keys(df):

    df = df.copy()

    df["blocking_keys"] = df.apply(
        create_blocking_keys,
        axis=1
    )

    return df


if __name__ == "__main__":

    data = pd.DataFrame({
        "entity_id": [
            "S1-001",
            "S1-002",
            "S1-003"
        ],

        "business_name": [
            "New Medical Clinic",
            "New Medical Clinic ABC",
            "Patho New Clinic"
        ],

        "business_address": [
            "12 MG Road Bhubaneswar Odisha",
            "12 MG Road Bhubaneswar Odisha",
            "45 Station Road Cuttack Odisha"
        ],

        "country": [
            "India",
            "India",
            "India"
        ]
    })

    data["name_normalized"] = (
        data["business_name"]
        .str.lower()
    )

    data["address_normalized"] = (
        data["business_address"]
        .str.lower()
    )

    result = add_blocking_keys(data)

    for _, row in result.iterrows():

        print("\n", row["entity_id"])
        print(row["business_name"])
        print(row["address_normalized"])

        print("Blocking keys:")

        for key in row["blocking_keys"]:
            print("  ", key)
