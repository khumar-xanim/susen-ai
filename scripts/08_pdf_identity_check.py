"""Does the first page of each leaflet mention the drug it is linked to? (finds wrong PDF links)"""
import json
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from config import CHUNKS_FILE, OUT_DIR, COL_ID, COL_NAME
from normalize import search_key

cat = pd.read_csv(OUT_DIR / "catalogue_clean.csv", dtype=str).fillna("")
brand = {r[COL_ID]: search_key(r[COL_NAME].split(",")[0]) for _, r in cat.iterrows()}
first = defaultdict(str)
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["chunk_index"] < 2:
            first[c["medicine_id"]] += " " + c["text"][:700]
rows = []
for med_id, text in first.items():
    b = brand.get(med_id, "")
    word = b.split()[0] if b.split() and len(b.split()[0]) >= 4 else b
    rows.append({"id": med_id, "brand": b, "score": fuzz.partial_ratio(word, search_key(text)), "first_text": text[:150]})
df = pd.DataFrame(rows).sort_values("score")
df.to_csv(OUT_DIR / "pdf_identity_check.csv", index=False, encoding="utf-8-sig")
bad = df[df["score"] < 85]
print(f"leaflets checked: {len(df)}; brand NOT found on the first page: {len(bad)}")
print(bad.head(25)[["id", "score"]].to_string(index=False))
print("full list: outputs\\pdf_identity_check.csv")