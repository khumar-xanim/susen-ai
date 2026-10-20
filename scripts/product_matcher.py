"""Find which catalogue product a user message is about (plan 1.4)."""
import re

import pandas as pd
from rapidfuzz import fuzz, process

from config import OUT_DIR, COL_ID, COL_NAME
from normalize import search_key

GENERIC = {"derman", "dermani", "dermanin", "dermanlar", "tablet", "kapsul", "ampula",
           "inyeksiya", "krem", "gel", "meshem", "mehlul", "hap"}


def _numbers(text):
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", str(text))}


class ProductIndex:
    def __init__(self, path=OUT_DIR / "catalogue_clean.csv"):
        df = pd.read_csv(path, dtype=str).fillna("")
        df["brand"] = df[COL_NAME].str.split(",").str[0].str.strip()
        df["brand_key"] = df["brand"].map(search_key)
        df["dose_nums"] = df["Dozası"].map(_numbers)
        self.df = df
        self.rows_by_key = {}      # full brand and first word of the brand -> row numbers
        for i, key in enumerate(df["brand_key"]):
            words = key.split()
            for k in {key, words[0] if words else ""}:
                if k:
                    self.rows_by_key.setdefault(k, set()).add(i)
        self.keys = list(self.rows_by_key)

    def find(self, query, cutoff=86, max_candidates=8):
        words = search_key(query).split()
        hits = {}                                   # key -> (score, number of words)
        for n in (4, 3, 2, 1):                      # every 1-4 word phrase of the message
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if len(gram) < 4 or (n == 1 and gram in GENERIC):
                    continue
                m = process.extractOne(gram, self.keys, scorer=fuzz.ratio, score_cutoff=cutoff)
                if m and (m[1], n) > hits.get(m[0], (0, 0)):
                    hits[m[0]] = (m[1], n)
        if not hits:
            return {"status": "not_found", "candidates": [], "n_total": 0}

        sure = {k: v for k, v in hits.items() if v[0] >= 92}
        if sure:                                    # confident: the longest phrase wins
            top = max(sure.values(), key=lambda v: (v[1], v[0]))
            keys = [k for k, v in sure.items() if v[1] == top[1] and v[0] >= top[0] - 2]
        else:                                       # only weak matches: best score wins
            top = max(hits.values())
            keys = [k for k, v in hits.items() if v[0] >= top[0] - 2]

        idx = set().union(*(self.rows_by_key[k] for k in keys))
        wanted = _numbers(query)                    # "abizol 10 mq" -> keep only the 10 mg row
        if len(idx) > 1 and wanted:
            idx = {i for i in idx if self.df.at[i, "dose_nums"] & wanted} or idx

        rows = self.df.loc[sorted(idx)].head(max_candidates)
        cands = [{"id": r[COL_ID], "name": r["brand"], "form": r["form_clean"], "dose": r["Dozası"],
                  "maker": r["İstehsalçı şirkət"], "price_azn": r["price_azn"],
                  "has_leaflet": r["has_leaflet"] == "True"} for _, r in rows.iterrows()]
        return {"status": "found" if len(idx) == 1 else "ambiguous",
                "candidates": cands, "n_total": len(idx)}