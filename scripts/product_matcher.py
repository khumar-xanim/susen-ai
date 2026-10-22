"""Find which catalogue product a user message or a photo is about (plan 1.4, 1.7)."""
import re
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

from config import OUT_DIR, COL_ID, COL_NAME
from normalize import search_key, skel

GENERIC = {"derman", "dermani", "dermanin", "dermanlar", "tablet", "tabletler", "kapsul", "ampula", "ampul", "flakon",
           "inyeksiya", "inyeksiyasi", "krem", "gel", "meshem", "melhem", "mehlul", "hap", "serbet", "damci", "sprey",
           "sakitlesdirici", "agrikesici", "antibiotik", "antibiotikler", "vitamin", "iyne", "siringa", "qelem"}
FORM_WORDS = {"tablet": "tablet", "tabletleri": "tablet", "kapsul": "kapsul", "inyeksiya": "inyeksiya",
              "inyeksiyasi": "inyeksiya", "ampula": "inyeksiya", "ampul": "inyeksiya", "flakon": "inyeksiya",
              "gel": "gel", "krem": "krem", "melhem": "melhem", "serbet": "serbet", "sirop": "serbet",
              "damci": "damci", "sprey": "sprey", "sham": "sam", "toz": "toz"}
SALTS = {"sulfate", "sodium", "hydrochloride", "maleate", "acid", "hcl", "tartrate", "citrate", "acetate",
         "phosphate", "potassium", "calcium", "bromide", "chloride", "succinate", "besylate", "mesylate"}
RU = dict(zip("абвгдеёжзийклмнопрстуфхцчшщыэюя",
              ["a", "b", "v", "g", "d", "e", "e", "j", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
               "u", "f", "x", "c", "c", "s", "s", "i", "e", "u", "a"]))


def numbers_of(text):
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", str(text))}


def _translit(text):
    return "".join(RU.get(c.lower(), c) if "\u0400" <= c <= "\u04ff" else c for c in text)


def _active_tokens(text):
    return tuple(sorted({w for w in search_key(text).split()
                         if len(w) >= 6 and w not in SALTS and not any(ch.isdigit() for ch in w)}))


class ProductIndex:
    def __init__(self, path=OUT_DIR / "catalogue_clean.csv"):
        df = pd.read_csv(path, dtype=str).fillna("")
        df["brand"] = df[COL_NAME].str.split(",").str[0].str.strip()
        df["brand_key"] = df["brand"].map(search_key)
        df["dose_nums"] = df["dose_clean"].map(numbers_of)
        df["form_key"] = df["form_clean"].map(search_key)
        df["active_key"] = df["active_clean"].map(_active_tokens)
        self.df = df
        self.rows_by_key = {}      # full brand and first word of the brand -> row numbers
        for i, key in enumerate(df["brand_key"]):
            words = key.split()
            for k in {key, words[0] if words else ""}:
                if k:
                    self.rows_by_key.setdefault(k, set()).add(i)
        self.keys = list(self.rows_by_key)
        self.skeys = [skel(k) for k in self.keys]

    def _cand(self, i):
        r = self.df.loc[i]
        return {"id": r[COL_ID], "name": r["brand"], "form": r["form_clean"], "dose": r["dose_clean"],
                "maker": r["maker_clean"], "price_azn": r["price_azn"], "has_leaflet": r["has_leaflet"] == "True"}

    def _pack(self, idx, max_candidates=8):
        idx = sorted(idx)
        if not idx:
            return {"status": "not_found", "candidates": [], "n_total": 0, "idx": [], "typo": None}
        return {"status": "found" if len(idx) == 1 else "ambiguous",
                "candidates": [self._cand(i) for i in idx[:max_candidates]], "n_total": len(idx), "idx": idx, "typo": None}

    def find(self, query, cutoff=86, max_candidates=8):
        words = search_key(_translit(query)).split()
        hits = {}                                    # brand key -> (score, words in phrase, phrase)
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if len(gram) < 4 or (n == 1 and gram in GENERIC):
                    continue
                m = process.extractOne(skel(gram), self.skeys, scorer=fuzz.ratio, score_cutoff=cutoff)
                if m:
                    key = self.keys[m[2]]
                    if (m[1], n) > hits.get(key, (0, 0, ""))[:2]:
                        hits[key] = (m[1], n, gram)
        if not hits:
            return {"status": "not_found", "candidates": [], "n_total": 0, "idx": [], "typo": None}
        sure = {k: v for k, v in hits.items() if v[0] >= 92}
        if sure:
            top = max(sure.values(), key=lambda v: (v[1], v[0]))
            keys = [k for k, v in sure.items() if v[1] == top[1] and v[0] >= top[0] - 2]
        else:
            top = max(hits.values())
            keys = [k for k, v in hits.items() if v[0] >= top[0] - 2]
        idx = set().union(*(self.rows_by_key[k] for k in keys))
        wanted = numbers_of(query)
        if len(idx) > 1 and wanted:                  # "abizol 10 mq" -> keep the 10 mg row
            idx = {i for i in idx if self.df.at[i, "dose_nums"] & wanted} or idx
        forms = {FORM_WORDS[w] for w in words if w in FORM_WORDS}
        if len(idx) > 1 and forms:                   # "adehep inyeksiya" -> keep the injection
            idx = {i for i in idx if any(f in self.df.at[i, "form_key"] for f in forms)} or idx
        out = self._pack(idx, max_candidates)
        if top[0] < 92:
            out["typo"] = {"said": top[2], "meant": self.df.at[out["idx"][0], "brand"], "score": top[0]}
        return out

    def find_all(self, query, cutoff=92):
        """Distinct brands named in one message (for example 'abizol ve adehep birlikde')."""
        words = search_key(_translit(query)).split()
        hits = {}
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if len(gram) < 4 or (n == 1 and gram in GENERIC):
                    continue
                m = process.extractOne(skel(gram), self.skeys, scorer=fuzz.ratio, score_cutoff=cutoff)
                if m and (m[1], n) > hits.get(self.keys[m[2]], (0, 0)):
                    hits[self.keys[m[2]]] = (m[1], n)
        groups, taken = [], set()
        for key, _ in sorted(hits.items(), key=lambda kv: (-kv[1][1], -kv[1][0])):
            rows = self.rows_by_key[key]
            if rows & taken:
                continue
            taken |= rows
            groups.append({"brand": self.df.at[min(rows), "brand"], "idx": sorted(rows)})
        return groups

    def identify_photo(self, brands, actives):
        """brands: cleaned words ordered by text size (biggest first); actives: ingredient words read from the photo."""
        res = self._pack([])
        for b in brands[:6]:
            r = self.find(b, cutoff=84)
            if r["status"] != "not_found":
                res = r
                break
        if actives:
            def n_hit(i):
                return sum(any(fuzz.ratio(a, t) >= 84 for t in self.df.at[i, "active_key"]) for a in actives)
            if res["status"] != "not_found":
                narrowed = [i for i in res["idx"] if n_hit(i) > 0]
                if narrowed:
                    best = max(n_hit(i) for i in narrowed)
                    res = self._pack([i for i in narrowed if n_hit(i) == best])
            else:
                scored = {i: n_hit(i) for i in range(len(self.df)) if self.df.at[i, "active_key"]}
                best = max(scored.values(), default=0)
                if best > 0:
                    res = self._pack([i for i, s in scored.items() if s == best and len(self.df.at[i, "active_key"]) <= best][:40])
        return res

    def identify(self, text, max_lines=25):
        """Drug from the first lines of a leaflet text (used by the OCR test)."""
        votes = Counter()
        for line in [x.strip() for x in text.splitlines() if x.strip()][:max_lines]:
            r = self.find(line)
            for i in r["idx"]:
                votes[i] += 1
        if not votes:
            return self._pack([])
        top = max(votes.values())
        return self._pack([i for i, v in votes.items() if v == top])

    def row(self, med_id):
        return self.df[self.df[COL_ID] == med_id].iloc[0]