"""Usage: python scripts\\show_chunks.py EXINASEYA   (ID prefix, first match is shown)"""
import json
import sys

from config import OUT_DIR

prefix = sys.argv[1].upper()
target = None
with open(OUT_DIR / "chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        target = target or (c["medicine_id"] if c["medicine_id"].startswith(prefix) else None)
        if c["medicine_id"] == target:
            print(f"--- [{c['chunk_index']}] {c['section']}  (p.{c['page_start']}-{c['page_end']}, {c['n_chars']} chars)")
            print(c["text"], "\n")