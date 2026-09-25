import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Unicode-aware text normalization.
    Preserves multilingual characters.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Keep:
    # L = letters
    # N = numbers
    # M = combining marks
    # whitespace
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_name(name):
    """
    Normalize business names.
    """
    return normalize_text(name)


def normalize_address(address):
    """
    Normalize business addresses.
    """
    if pd.isna(address):
        return ""

    text = normalize_text(address)

    # Common address abbreviations
    replacements = {
        r"\bst\b": "street",
        r"\brd\b": "road",
        r"\bave\b": "avenue",
        r"\bav\b": "avenue",
        r"\bdr\b": "drive",
        r"\bln\b": "lane",
        r"\bblvd\b": "boulevard",
        r"\bhwy\b": "highway",
        r"\bct\b": "court",
        r"\bapt\b": "apartment",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


if __name__ == "__main__":

    names = [
        "B+ Retail Inc",
        "McDonald's Restaurant Ltd.",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Pvt. EFS Print Ventures Ltd."
    ]

    addresses = [
        "1795 Westchester Drive, High Point, NC",
        "105 ELM ST, MORGANTON, NC",
        "Door No 183, 41St Cross, 22Nd Main"
    ]

    print("NAME NORMALIZATION")

    for name in names:
        print(name, "→", normalize_name(name))

    print("\nADDRESS NORMALIZATION")

    for address in addresses:
        print(address, "→", normalize_address(address))