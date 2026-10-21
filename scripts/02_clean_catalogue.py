"""Step 2 - clean catalogue (plan 1.1). Original columns are kept; new columns are added."""
import pandas as pd
from catalogue_names import reconcile
from config import (EXCEL, OUT_DIR, REPORT_DIR, EXPECTED_COLUMNS,
                    COL_ID, COL_NAME, COL_PRICE, COL_FORM, COL_PDF)

# Typos found in the Step 1 audit
FORM_FIXES = {"KAPSUL": "Kapsul", "İnfuzya üçün məhlul": "İnfuziya üçün məhlul"}
# Only a flag for human review. The price itself is never changed.
PRICE_REVIEW_ABOVE = 1000  # AZN

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


df = pd.read_excel(EXCEL, engine="openpyxl", dtype=str)
df.columns = [c.strip() for c in df.columns]

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise SystemExit(f"Columns missing from the Excel file: {missing}")

# 1. Trim spaces in every text cell; blank cells become real empty values
df = df.apply(lambda col: col.str.strip()).replace("", pd.NA)

# 2. Reconcile the table columns with the product string; if they disagree, the string wins
rec = df.apply(lambda r: pd.Series(reconcile(r[COL_NAME], r[COL_FORM], r["Təsiredici maddənin adı"], r["Dozası"],
                                             r["İstehsalçı şirkət"], r["Ölkə"], r["Ticarət qablaşdırması"], r["Miqdar"])),
               axis=1)
df = pd.concat([df, rec], axis=1)
df["form_clean"] = df["form_clean"].replace(FORM_FIXES)

# 3. Price as a number in a NEW column, plus a review flag
df["price_azn"] = pd.to_numeric(df[COL_PRICE].str.replace(",", ".", regex=False), errors="coerce")
df["price_review"] = df["price_azn"] > PRICE_REVIEW_ABOVE

# 4. Does this drug have a leaflet? (1 = yes, 0 = no)
df["has_leaflet"] = df[COL_PDF] == "1"

# 5. Which catalogue fields are empty in each row (the chatbot can say "no information")
check_cols = [c for c in EXPECTED_COLUMNS if c not in ("ID", "Sıra №")]
df["missing_fields"] = df[check_cols].isna().apply(lambda r: "; ".join(r.index[r]), axis=1)

# ---------- report ----------
log(f"rows: {len(df)}")
log("\nform_clean values:")
log(df["form_clean"].value_counts(dropna=False).to_string())
log(f"\nprice_azn not a number: {df['price_azn'].isna().sum()}")
log(f"price_review (> {PRICE_REVIEW_ABOVE} AZN): {df['price_review'].sum()}")
short = df.loc[df["price_review"], [COL_ID, "price_azn"]].sort_values("price_azn", ascending=False)
log(short.to_string(index=False))
log(f"\nrows with a leaflet: {df['has_leaflet'].sum()}, without: {(~df['has_leaflet']).sum()}")
log("\nmost common missing-field combinations:")
log(df.loc[df["missing_fields"] != "", "missing_fields"].value_counts().head(10).to_string())

bad = df[~df["cols_match_name"].astype(bool)]
log(f"\nrows where table columns disagree with the product string: {len(bad)}")
log(bad["mismatch_fields"].value_counts().head(10).to_string())
log(f"product strings that could not be parsed: {(df['mismatch_fields'] == 'unparsed').sum()}")
bad.to_csv(OUT_DIR / "name_column_mismatch.csv", index=False, encoding="utf-8-sig")

df.to_csv(OUT_DIR / "catalogue_clean.csv", index=False, encoding="utf-8-sig")
(REPORT_DIR / "02_clean_report.txt").write_text("\n".join(lines), encoding="utf-8")
log(f"\nSaved: {OUT_DIR / 'catalogue_clean.csv'}")