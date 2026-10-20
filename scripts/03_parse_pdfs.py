"""Step 3 - extract text from leaflet PDFs and chunk by bold headings."""
import argparse
import json
import os
from collections import Counter
from datetime import date
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from config import PDF_DIR, OUT_DIR, REPORT_DIR, COL_ID, COL_NAME
from leaflet_parser import parse_pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N PDFs (quick test)")
    args = ap.parse_args()

    clean_csv = OUT_DIR / "catalogue_clean.csv"
    if not clean_csv.exists():
        raise SystemExit("outputs/catalogue_clean.csv missing - run Step 2 first.")
    cat = pd.read_csv(clean_csv, dtype=str)
    names = dict(zip(cat[COL_ID], cat[COL_NAME].str.split(",").str[0].str.strip()))

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    workers = max(1, (os.cpu_count() or 2) - 2)
    today = date.today().isoformat()
    print(f"{len(pdfs)} PDFs, {workers} workers")

    logs, vocab = [], Counter()
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as out, Pool(workers) as pool:
        for log, chunks, headings in tqdm(pool.imap(parse_pdf_path, map(str, pdfs), chunksize=8),
                                          total=len(pdfs)):
            log["in_catalogue"] = log["id"] in names
            logs.append(log)
            vocab.update(headings)
            if not log["in_catalogue"]:
                continue
            for c in chunks:
                c["drug_name"] = names[c["medicine_id"]]
                c["extracted_at"] = today
                out.write(json.dumps(c, ensure_ascii=False) + "\n")

    df = pd.DataFrame(logs)
    df.to_csv(OUT_DIR / "pdf_extraction_log.csv", index=False, encoding="utf-8-sig")
    vdf = pd.DataFrame(vocab.most_common(), columns=["heading", "count"])
    vdf.to_csv(OUT_DIR / "heading_vocabulary.csv", index=False, encoding="utf-8-sig")

    lines = [f"PDFs processed: {len(df)}", "", "status:", df["status"].value_counts().to_string()]
    lines += ["", "PDFs not in catalogue: " + ", ".join(df.loc[~df["in_catalogue"], "file"]) or "none"]
    ok = df[df["n_chunks"] > 0]
    lines += ["", f"chunks total: {df['n_chunks'].sum()}",
              "chunks per PDF:", ok["n_chunks"].describe().round(1).to_string(),
              "chunk size (chars):",
              pd.Series([0]).describe().to_string() if ok.empty else
              (ok["chars"] / ok["n_chunks"]).describe().round(0).to_string() + "   <- mean size per PDF"]
    no_head = df[(df["n_chunks"] > 0) & (df["n_headings"] == 0)]
    lines += ["", f"PDFs with text but NO bold headings found: {len(no_head)}",
              ", ".join(no_head["id"].head(15))]
    lines += ["", f"needs OCR (no text at all): {(df['status'] == 'needs_ocr').sum()}",
              f"partly needs OCR: {(df['status'] == 'partial_ocr').sum()}",
              ", ".join(df.loc[df['status'].isin(['needs_ocr', 'partial_ocr']), 'id'].head(10))]
    if (df["status"] == "error").any():
        lines += ["", "errors:", df.loc[df["status"] == "error", ["file", "error"]].head(10).to_string(index=False)]
    top = vdf.head(40)
    lines += ["", f"distinct headings: {len(vdf)}; top 40 cover "
              f"{top['count'].sum() / max(vdf['count'].sum(), 1):.0%} of all heading occurrences",
              top.to_string(index=False)]
    report = "\n".join(lines)
    print("\n" + report)
    (REPORT_DIR / "03_parse_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":      # required on Windows for multiprocessing
    main()