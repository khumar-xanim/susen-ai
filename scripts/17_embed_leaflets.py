"""Embed every chunk with bge-m3 (one time). Usage: python scripts\\17_embed_leaflets.py [--source leaflets|us] [--limit N]
Leaflets: about 90 minutes. Keep the laptop plugged in and close heavy programs. Ctrl+C is safe: finished parts are kept.
Run it AFTER 03_parse_pdfs.py and 09_ocr_scans.py (they change chunks.jsonl)."""
import argparse
import json

import numpy as np
from tqdm import tqdm

from config import CHUNKS_FILE, OUT_DIR
from kb import embed

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["leaflets", "us"], default="leaflets")
ap.add_argument("--limit", type=int, default=0, help="quick test on the first N chunks (does not write the final files)")
args = ap.parse_args()
src = CHUNKS_FILE if args.source == "leaflets" else OUT_DIR / "us_chunks.jsonl"
prefix = "leaflet" if args.source == "leaflets" else "us"
rows = [json.loads(x) for x in open(src, encoding="utf-8")]
if args.limit:
    rows = rows[:args.limit]
texts = [f"{r['section']}. {r['text']}" for r in rows]
parts = OUT_DIR / f"{prefix}_parts{'_test' if args.limit else ''}"
parts.mkdir(exist_ok=True)
SHARD = 1000
for start in tqdm(range(0, len(texts), SHARD), desc="parts"):
    path = parts / f"{start:07d}.npy"
    if not path.exists():
        np.save(path, embed(texts[start:start + SHARD]).astype(np.float16))
if args.limit:
    print("test run finished; delete the *_parts_test folder")
else:
    vecs = np.vstack([np.load(p) for p in sorted(parts.glob("*.npy"))])
    assert len(vecs) == len(rows), (len(vecs), len(rows))
    np.save(OUT_DIR / f"{prefix}_vectors.npy", vecs)
    with open(OUT_DIR / f"{prefix}_meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["medicine_id"], "i": r["chunk_index"]}) + "\n")
    print("saved", vecs.shape)