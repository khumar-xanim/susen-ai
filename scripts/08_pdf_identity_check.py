"""Does the beginning of each leaflet mention the drug it is linked to? (finds wrong PDF links)"""
import json
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from config import CHUNKS_FILE, OUT_DIR, COL_ID, COL_NAME
from normalize import search_key

TR = str.maketrans({"c": "k", "w": "v"})          # Ozempic ~ Ozempik
cat = pd.read_csv(OUT_DIR / "catalogue_clean.csv", dtype=str).fillna("")
brand = {r[COL_ID]: search_key(r[COL_NAME].split(",")[0]) for _, r in cat.iterrows()}
first = defaultdict(str)
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["chunk_index"] < 6:
            first[c["medicine_id"]] += " " + c["text"][:800]
rows = []
for med_id, text in first.items():
    b = brand.get(med_id, "")
    word = b.split()[0] if b.split() and len(b.split()[0]) >= 4 else b
    t = search_key(text).translate(TR)
    score = max(fuzz.partial_ratio(word.translate(TR), t), fuzz.partial_ratio(b.translate(TR), t))
    rows.append({"id": med_id, "brand": b, "score": score, "first_text": text[:150]})
df = pd.DataFrame(rows).sort_values("score")
df.to_csv(OUT_DIR / "pdf_identity_check.csv", index=False, encoding="utf-8-sig")
bad = df[df["score"] < 85]
print(f"leaflets checked: {len(df)}; brand NOT found near the start: {len(bad)}")
print(bad.head(25)[["id", "score"]].to_string(index=False))
print("full list: outputs\\pdf_identity_check.csv (check a few in the chat window with SUSEN_ADMIN=1)")