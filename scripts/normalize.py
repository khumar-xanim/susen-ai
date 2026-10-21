"""Azerbaijani text normalisation (plan 1.2). search_key() gives the form used for matching."""
import re
import unicodedata

# Cyrillic letters that look like Latin ones (found in some leaflet headings, e.g. "İstifаdə qаydаsı")
CYR = str.maketrans({
    "а": "a", "А": "a", "е": "e", "Е": "e", "о": "o", "О": "o", "р": "p", "Р": "p", "с": "c", "С": "c",
    "х": "x", "Х": "x", "у": "y", "У": "y", "і": "i", "І": "i", "к": "k", "К": "k", "М": "m", "Т": "t",
    "Н": "h", "В": "b", "ә": "ə", "Ә": "Ə", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü", "һ": "h", "Һ": "h", "ј": "j",
})
# Python's own lower() would turn İ into "i" + a hidden dot, so Azerbaijani letters are folded first.
FOLD = str.maketrans({
    "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})


def search_key(text):
    text = unicodedata.normalize("NFC", str(text)).translate(CYR).translate(FOLD)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[\W_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()