"""Step 1 - catalogue audit and PDF folder inspection (plan section 1.1)."""
from collections import Counter

import pandas as pd

from config import (EXCEL, PDF_DIR, OUT_DIR, REPORT_DIR, COL_ID, COL_NAME,
                    COL_PRICE, COL_FORM, COL_INTAKE, COL_COUNTRY, COL_PDF)

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def section(title):
    log("\n" + "=" * 60)
    log(title)


def save(frame, name):
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


if not EXCEL.exists():
    raise SystemExit(f"Excel file not found: {EXCEL}")

# dtype=str keeps every cell as text, so nothing is silently converted.
df = pd.read_excel(EXCEL, engine="openpyxl", dtype=str)
df.columns = [c.strip() for c in df.columns]

section("1. SHAPE")
log(f"rows: {len(df)}, columns: {len(df.columns)}")

section("2. DUPLICATES")
log(f"rows with a duplicated ID: {df[COL_ID].duplicated(keep=False).sum()}")
log(f"rows with a duplicated product string: {df[COL_NAME].duplicated(keep=False).sum()}")

section("3. EMPTY CELLS PER COLUMN")
empty = df.apply(lambda c: c.isna() | (c.str.strip() == ""))
log(empty.sum().to_string())
save(df[empty.any(axis=1)], "rows_with_empty_cells.csv")

section("4. PRICE (AZN)")
price = pd.to_numeric(df[COL_PRICE].str.replace(",", ".", regex=False), errors="coerce")
log(f"missing / non-numeric / <= 0: {(price.isna() | (price <= 0)).sum()}")
log(price.describe().round(2).to_string())
view = df.assign(price=price)[[COL_ID, COL_NAME, "price"]].copy()
view[COL_NAME] = view[COL_NAME].str.slice(0, 70)
log("\n10 highest prices:")
log(view.nlargest(10, "price").to_string(index=False))
log("\n5 lowest prices:")
log(view.nsmallest(5, "price").to_string(index=False))
save(df.assign(price=price).nlargest(50, "price"), "top50_prices.csv")

section("5. CATEGORICAL VALUES")
for col in [COL_FORM, COL_INTAKE, COL_COUNTRY, COL_PDF]:
    log(f"\n--- {col} ---")
    log(df[col].value_counts(dropna=False).to_string())
forms = df[COL_FORM].str.strip()
variants = forms.groupby(forms.str.casefold()).unique()
log("\nforms spelled in more than one way (case only):")
log(variants[variants.apply(len) > 1].to_string())
rare = forms.value_counts()
rare = rare[rare < 5]
log("\nrare forms (< 5 rows) - check for typos:")
log(rare.to_string())
save(df[forms.isin(rare.index)], "rare_form_rows.csv")

section("6. PDF FOLDER")
if not PDF_DIR.exists():
    log(f"PDF folder NOT FOUND: {PDF_DIR}")
else:
    files = [p for p in PDF_DIR.rglob("*") if p.is_file()]
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    log(f"folder: {PDF_DIR}")
    log(f"files: {len(files)}, PDFs: {len(pdfs)}")
    log(f"extensions: {dict(Counter(p.suffix.lower() for p in files))}")
    log(f"total size: {sum(p.stat().st_size for p in files) / 1024**2:.0f} MB")
    log("first 10 PDF names:")
    for p in sorted(pdfs)[:10]:
        log("   " + p.name)

    section("7. PDF CROSS-CHECK (hypothesis: file name = ID + .pdf)")
    stems = {p.stem for p in pdfs}
    ids = set(df[COL_ID])
    flag1 = set(df.loc[df[COL_PDF] == "1", COL_ID])
    flag0 = set(df.loc[df[COL_PDF] == "0", COL_ID])
    log(f"catalogue IDs whose name matches a PDF: {len(ids & stems)}")
    log(f"flagged 1 but no matching file: {len(flag1 - stems)}")
    log(f"flagged 0 but a matching file exists: {len(flag0 & stems)}")
    log(f"PDF files that match no catalogue ID: {len(stems - ids)}")
    pd.Series(sorted(flag1 - stems), name=COL_ID).to_csv(OUT_DIR / "flag1_missing_pdf.csv", index=False)
    pd.Series(sorted(stems - ids), name="pdf_stem").to_csv(OUT_DIR / "unmatched_pdfs.csv", index=False)

(REPORT_DIR / "01_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nReport saved: {REPORT_DIR / '01_audit_report.txt'}")