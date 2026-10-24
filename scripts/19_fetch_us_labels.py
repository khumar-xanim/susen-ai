"""US FDA labels (openFDA) for catalogue drugs that have NO Azerbaijani leaflet. Usage: python scripts\\19_fetch_us_labels.py [--limit 30]
Optional: $env:OPENFDA_API_KEY="..." (free key). Without a key openFDA allows about 1000 requests per day; the cache lets you continue tomorrow."""
import argparse
import json
import os
import re
import time

import requests
from tqdm import tqdm

from config import COL_ID, OUT_DIR
from leaflet_parser import chunk_section
from product_matcher import ProductIndex

API = "https://api.fda.gov/drug/label.json"
CACHE = OUT_DIR / "us_labels_cache.json"
SYN = {"paracetamol": "acetaminophen", "salbutamol": "albuterol", "adrenaline": "epinephrine",
       "noradrenaline": "norepinephrine", "aciclovir": "acyclovir", "ciclosporin": "cyclosporine",
       "rifampicin": "rifampin", "glyceryl": "nitroglycerin", "trinitrate": "nitroglycerin"}
FIELDS = {"indications_and_usage": "indications", "contraindications": "contraindications",
          "dosage_and_administration": "dosage", "adverse_reactions": "side_effects", "drug_interactions": "interactions",
          "pregnancy": "pregnancy", "pediatric_use": "pediatric", "overdosage": "overdose",
          "storage_and_handling": "storage", "how_supplied": "storage", "warnings_and_cautions": "special",
          "warnings": "special", "clinical_pharmacology": "pharmacology", "description": "composition",
          "mechanism_of_action": "pharmacology"}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
key = os.getenv("OPENFDA_API_KEY")
delay = 0.5 if key else 1.7
df = ProductIndex().df
todo = df[df["has_leaflet"] != "True"]
cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}


def tokens(row):
    toks = list(dict.fromkeys(SYN.get(t, t) for t in row["active_key"]))
    return toks if 1 <= len(toks) <= 3 else []        # skip herbal/homeopathic/multi-ingredient products


def fetch(toks):
    search = "+AND+".join(f'openfda.generic_name:"{t}"' for t in toks)
    r = requests.get(f"{API}?search={search}&limit=5" + (f"&api_key={key}" if key else ""), timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        raise SystemExit("openFDA daily limit reached: run this script again tomorrow, or use a free API key.")
    r.raise_for_status()
    best, best_n = None, 0
    for res in r.json().get("results", []):
        n = sum(bool(res.get(f)) for f in FIELDS) + 3 * ("HUMAN PRESCRIPTION DRUG" in (res.get("openfda", {}).get("product_type") or []))
        if n > best_n:
            best, best_n = res, n
    if not best:
        return None
    gen = ", ".join((best.get("openfda", {}).get("generic_name") or [])[:2])
    return {"generic": gen, "sections": {f: " ".join(best[f])[:6000] for f in FIELDS if best.get(f)}}


wanted = {}
for _, row in todo.iterrows():
    t = tokens(row)
    if t:
        wanted.setdefault("|".join(t), t)
for k, t in tqdm(list(wanted.items())[: args.limit or None], desc="labels"):
    if k not in cache:
        cache[k] = fetch(t)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(delay)

n_drugs, rows = 0, []
for _, row in todo.iterrows():
    lab = cache.get("|".join(tokens(row))) if tokens(row) else None
    if not lab:
        continue
    n_drugs += 1
    idx = 0
    for field, text in lab["sections"].items():
        text = " ".join(re.sub(r"\s+", " ", text).split())
        for ch in chunk_section({"heading": field, "paras": [(0, 0, text)]}):
            rows.append({"chunk_id": f"{row[COL_ID]}#us{idx:03d}", "medicine_id": row[COL_ID],
                         "section": field.replace("_", " ").capitalize(), "category": FIELDS[field],
                         "chunk_index": idx, "text": ch["text"], "lang": "en", "generic": lab["generic"]})
            idx += 1
with open(OUT_DIR / "us_chunks.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"drugs without a leaflet: {len(todo)}; got a US label for {n_drugs}; chunks: {len(rows)}")