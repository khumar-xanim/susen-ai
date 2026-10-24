"""Dense (embedding) search INSIDE one drug's chunks (plan 1.9, 1.10) and the US label chunks."""
import json
from collections import defaultdict

import numpy as np

from config import OUT_DIR


class DenseStore:
    def __init__(self, prefix):
        vec, meta = OUT_DIR / f"{prefix}_vectors.npy", OUT_DIR / f"{prefix}_meta.jsonl"
        self.ok = vec.exists() and meta.exists()
        self.rows = defaultdict(dict)        # medicine_id -> {chunk_index: row in the matrix}
        if self.ok:
            self.vecs = np.load(vec, mmap_mode="r")
            for r, line in enumerate(open(meta, encoding="utf-8")):
                m = json.loads(line)
                self.rows[m["id"]][m["i"]] = r

    def scores(self, med_id, qvec):
        rows = self.rows.get(med_id)
        if not rows:
            return {}
        sims = np.asarray(self.vecs[list(rows.values())], dtype=np.float32) @ qvec
        return dict(zip(rows.keys(), map(float, sims)))


LEAFLET_DENSE = DenseStore("leaflet")
US_DENSE = DenseStore("us")


def load_us_chunks():
    out, path = defaultdict(list), OUT_DIR / "us_chunks.jsonl"
    if path.exists():
        for line in open(path, encoding="utf-8"):
            c = json.loads(line)
            out[c["medicine_id"]].append({"section": c["section"], "page_start": 0, "page_end": 0, "text": c["text"],
                                          "chunk_index": c["chunk_index"], "category": c["category"], "lang": "en"})
    return out