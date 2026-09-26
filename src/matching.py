import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# TEXT UTILITIES
# ============================================================

def tokenize(text):
    if not text:
        return set()

    return set(str(text).lower().split())


def normalize_for_matching(text):
    if text is None:
        return ""

    text = str(text).lower()

    # Keep letters/numbers from all scripts
    text = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in text
    )

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def token_jaccard(a, b):
    """
    Jaccard similarity between token sets.
    """
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return intersection / union if union else 0.0


def sequence_similarity(a, b):
    """
    Character-level similarity.
    """

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def substring_score(a, b):
    """
    Checks whether one string occurs inside the other.
    """

    if not a or not b:
        return 0.0

    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))

        if longer == 0:
            return 0.0

        return shorter / longer

    return 0.0


def number_overlap(a, b):
    """
    Compare numbers appearing in addresses.
    """

    nums_a = set(
        re.findall(r"\b\d+[a-z]?\b", a)
    )

    nums_b = set(
        re.findall(r"\b\d+[a-z]?\b", b)
    )

    if not nums_a and not nums_b:
        return 1.0

    if not nums_a or not nums_b:
        return 0.0

    return len(nums_a & nums_b) / len(nums_a | nums_b)


# ============================================================
# PAIR FEATURES
# ============================================================

def extract_pair_features(s1, candidate):
    """
    Create numerical similarity features for one S1-S2/S3 pair.

    Returns a numpy array.
    """

    s1_name = normalize_for_matching(
        s1.get("business_name", "")
    )

    candidate_name = normalize_for_matching(
        candidate.get("business_name", "")
    )

    s1_address = normalize_for_matching(
        s1.get("business_address", "")
    )

    candidate_address = normalize_for_matching(
        candidate.get("business_address", "")
    )

    s1_country = normalize_for_matching(
        s1.get("country", "")
    )

    candidate_country = normalize_for_matching(
        candidate.get("country", "")
    )


    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_jaccard = token_jaccard(
        s1_name,
        candidate_name
    )

    name_sequence = sequence_similarity(
        s1_name,
        candidate_name
    )

    name_substring = substring_score(
        s1_name,
        candidate_name
    )


    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_jaccard = token_jaccard(
        s1_address,
        candidate_address
    )

    address_sequence = sequence_similarity(
        s1_address,
        candidate_address
    )

    address_numbers = number_overlap(
        s1_address,
        candidate_address
    )


    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = float(
        s1_country != ""
        and s1_country == candidate_country
    )


    # --------------------------------------------------------
    # EXACT MATCH SIGNALS
    # --------------------------------------------------------

    exact_name = float(
        s1_name != ""
        and s1_name == candidate_name
    )

    exact_address = float(
        s1_address != ""
        and s1_address == candidate_address
    )


    return np.array([
        name_jaccard,
        name_sequence,
        name_substring,
        address_jaccard,
        address_sequence,
        address_numbers,
        country_match,
        exact_name,
        exact_address,
    ], dtype=np.float32)


# ============================================================
# MODEL
# ============================================================

def create_model():
    """
    Logistic Regression model.

    StandardScaler + LogisticRegression makes the model
    easy to train and use for match probabilities.
    """

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


def train_model(X, y):
    """
    Train the matching model.

    X = feature matrix
    y = 0/1 labels
    """

    model = create_model()

    model.fit(X, y)

    return model


def predict_match_probability(model, features):
    """
    Return probability that a pair is a true match.
    """

    features = np.asarray(features)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return model.predict_proba(features)[:, 1]


# ============================================================
# SIMPLE RULE-BASED SCORE
# ============================================================

def rule_based_score(s1, candidate):
    """
    Useful baseline before the ML model.

    This is NOT the final decision rule.
    """

    features = extract_pair_features(
        s1,
        candidate
    )

    (
        name_jaccard,
        name_sequence,
        name_substring,
        address_jaccard,
        address_sequence,
        address_numbers,
        country_match,
        exact_name,
        exact_address,
    ) = features

    score = (
        0.30 * name_jaccard
        + 0.20 * name_sequence
        + 0.10 * name_substring
        + 0.20 * address_jaccard
        + 0.10 * address_sequence
        + 0.05 * address_numbers
        + 0.05 * country_match
    )

    if exact_name:
        score += 0.20

    if exact_address:
        score += 0.20

    return min(score, 1.0)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    s1 = {
        "business_name": "ABC Technologies Pvt Ltd",
        "business_address": "12 MG Road Bhubaneswar Odisha",
        "country": "India"
    }

    candidate = {
        "business_name": "ABC Technologies Private Limited",
        "business_address": "12 MG Rd Bhubaneswar Odisha",
        "country": "India"
    }

    features = extract_pair_features(
        s1,
        candidate
    )

    print("Pair features:")
    print(features)

    print(
        "\nRule-based score:",
        rule_based_score(s1, candidate)
    )
