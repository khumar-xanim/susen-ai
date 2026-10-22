"""Simulated phone photos of test-split leaflets: OCR accuracy (plan 1.7). Usage: python scripts\\13_eval_ocr.py [--n 15]"""
import argparse
import random

import cv2
import numpy as np
import pandas as pd
import pymupdf
from rapidfuzz.distance import Levenshtein

from config import EVAL_DIR, PDF_DIR, COL_ID, split_of
from normalize import search_key
from ocr import ocr_array
from product_matcher import ProductIndex

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
index = ProductIndex()
ids = [i for i in index.df[COL_ID] if split_of(i) == "test" and (PDF_DIR / f"{i}.pdf").exists()]
random.Random(args.seed).shuffle(ids)


def degrade(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4))
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.1), beta=rng.uniform(-40, 10))
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 70))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


rows = []
for med_id in ids:
    if len(rows) >= args.n:
        break
    with pymupdf.open(PDF_DIR / f"{med_id}.pdf") as doc:
        page = doc.load_page(0)
        truth = " ".join(page.get_text().split())
        if len(truth) < 500:
            continue
        pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    raw = ocr_array(degrade(img))
    text = " ".join(raw.split())
    res = index.identify(raw)
    rows.append({"id": med_id, "cer_raw": Levenshtein.normalized_distance(truth, text),
                 "cer_folded": Levenshtein.normalized_distance(search_key(truth), search_key(text)),
                 "drug_identified": med_id in {c["id"] for c in res["candidates"]}})
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(EVAL_DIR / "ocr_eval.csv", index=False)
print(f"\npages: {len(df)}  mean CER (raw text): {df['cer_raw'].mean():.1%}  "
      f"mean CER (ignoring ə/ş/ç...): {df['cer_folded'].mean():.1%}  drug identified: {df['drug_identified'].mean():.0%}")