"""Are the table columns shifted relative to the product strings? For each disagreeing row, find the row whose
product string contains its columns. Usage: python scripts\\14_alignment_audit.py"""
from collections import Counter

import pandas as pd

from config import OUT_DIR, COL_ID, COL_NAME
from normalize import search_key

df = pd.read_csv(OUT_DIR / "catalogue_clean.csv", dtype=str).fillna("")
n = len(df)
names = [search_key(x) for x in df[COL_NAME]]
act = [search_key(x) for x in df["Təsiredici maddənin adı"]]
mk = [search_key(x) for x in df["İstehsalçı şirkət"]]
bad = [i for i in range(n) if df.at[i, "cols_match_name"] == "False"]
print(f"rows: {n}; table columns disagree with the product string: {len(bad)}")
rows, offsets = [], Counter()
for i in bad:
    cand = [j for j in range(n) if act[i] and act[i] in names[j] and mk[i] in names[j]]
    best = min(cand, key=lambda j: abs(j - i)) if cand else None
    rows.append({"row": i, COL_ID: df.at[i, COL_ID], "candidates": len(cand), "offset": (best - i) if best is not None else ""})
    if len(cand) == 1:
        offsets[best - i] += 1
out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "alignment_audit.csv", index=False)
print(f"columns found in some product string: {(out['candidates'] > 0).sum()}; in exactly one: {(out['candidates'] == 1).sum()}")
print("most common row offsets (unique matches):", offsets.most_common(10))
uniq = out[out["candidates"] == 1].copy()
uniq["offset"] = uniq["offset"].astype(int)
uniq["block"] = uniq["row"] // 250 * 250
print("\nmedian offset per block of 250 rows:")
print(uniq.groupby("block")["offset"].agg(["count", "median"]).to_string())