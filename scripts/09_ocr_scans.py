"""OCR the leaflet pages that have no usable text layer and add them to chunks.jsonl (plan 1.7)."""
import json
from datetime import date

import cv2
import numpy as np
import pandas as pd
import pymupdf

from config import CHUNKS_FILE, OUT_DIR, PDF_DIR, COL_ID, COL_NAME
from ocr import ocr_array
from retrieval import split_by_headings

log = pd.read_csv(OUT_DIR / "pdf_extraction_log.csv", dtype=str).fillna("")
targets = log[(log["ocr_pages"] != "") & (log["in_catalogue"] == "True")]
cat = pd.read_csv(OUT_DIR / "catalogue_clean.csv", dtype=str).fillna("")
names = dict(zip(cat[COL_ID], cat[COL_NAME].str.split(",").str[0].str.strip()))
ids = set(targets["id"])
print("pages to OCR:", {r["id"]: r["ocr_pages"] for _, r in targets.iterrows()})

text_dir = OUT_DIR / "ocr_text"
text_dir.mkdir(exist_ok=True)
kept, last_index = [], {}
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["medicine_id"] in ids:
            if c.get("ocr"):
                continue                                   # old OCR chunks are redone
            last_index[c["medicine_id"]] = max(last_index.get(c["medicine_id"], -1), c["chunk_index"])
        kept.append(line)

new = []
for _, r in targets.iterrows():
    idx = last_index.get(r["id"], -1) + 1
    with pymupdf.open(PDF_DIR / r["file"]) as doc:
        for p in [int(x) for x in r["ocr_pages"].split(",")]:
            pix = doc.load_page(p - 1).get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            text = ocr_array(img)
            (text_dir / f"{r['id']}_p{p}.txt").write_text(text, encoding="utf-8")
            n_before = len(new)
            for ch in split_by_headings(text, page=p):
                new.append({"chunk_id": f"{r['id']}#ocr{idx:03d}", "medicine_id": r["id"], "drug_name": names.get(r["id"], ""),
                            "section": ch["section"], "page_start": ch["page_start"], "page_end": ch["page_end"],
                            "chunk_index": idx, "text": ch["text"], "n_chars": len(ch["text"]),
                            "source_file": r["file"], "extracted_at": date.today().isoformat(), "ocr": True})
                idx += 1
            print(f"{r['id']} p.{p}: {len(text)} characters, {len(new) - n_before} chunks")

tmp = CHUNKS_FILE.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    f.writelines(kept)
    for c in new:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
tmp.replace(CHUNKS_FILE)
print(f"added {len(new)} OCR chunks. Read the text in outputs\\ocr_text\\ and compare it with the PDF pages.")