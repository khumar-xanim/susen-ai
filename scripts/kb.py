"""Knowledge base of trusted background texts (plan 1.9, 1.10): dense retrieval with bge-m3 through Ollama."""
import json

import numpy as np
import ollama

from config import OUT_DIR
from normalize import search_key

EMBED_MODEL = "bge-m3"
KB_CHUNKS = OUT_DIR / "kb_chunks.jsonl"
KB_VECTORS = OUT_DIR / "kb_vectors.npy"
MIN_SCORE = 0.45        # starting value: tune it with scripts\16_kb_search.py

def embed(texts, batch=16, cpu=False):
    opts = {"num_gpu": 0} if cpu else None
    out = []
    for i in range(0, len(texts), batch):
        res = ollama.embed(model=EMBED_MODEL, input=[search_key(t)[:2000] or "bos" for t in texts[i:i + batch]],
                           options=opts, keep_alive="30m")
        out.extend(res["embeddings"])
    arr = np.asarray(out, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

class KnowledgeBase:
    def __init__(self):
        self.ok = KB_CHUNKS.exists() and KB_VECTORS.exists()
        if self.ok:
            self.meta = [json.loads(x) for x in open(KB_CHUNKS, encoding="utf-8")]
            self.vecs = np.load(KB_VECTORS)

    def search(self, query, k=4, min_score=MIN_SCORE):
        if not self.ok:
            return []
        try:
            q = embed([query], cpu=True)[0]
        except Exception:
            return []
        scores = self.vecs @ q
        return [{**self.meta[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k] if scores[i] >= min_score]