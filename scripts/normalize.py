"""Azerbaijani text normalisation (plan 1.2). search_key() gives the form used for matching."""
import re
import unicodedata

# Letters first, uppercase included. Python's own lower() would turn İ into "i" + a hidden dot.
FOLD = str.maketrans({
    "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})


def search_key(text):
    text = unicodedata.normalize("NFC", str(text)).translate(FOLD)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[\W_]+", " ", text)   # punctuation, ®, № become spaces
    return re.sub(r"\s+", " ", text).strip()