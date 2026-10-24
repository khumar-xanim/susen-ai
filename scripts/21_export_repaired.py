"""Export the catalogue with the repaired detail columns (to report the problem to the data owners)."""
import pandas as pd

from config import OUT_DIR

df = pd.read_csv(OUT_DIR / "catalogue_clean.csv", dtype=str).fillna("")
cols = ["ID", "Sıra №", "Dərmanın adı", "Qiymət (AZN)", "form_clean", "active_clean", "dose_clean", "maker_clean",
        "country_clean", "pack_clean", "qty_clean", "cols_match_name", "mismatch_fields", "PDF_Y_N"]
df[cols].to_excel(OUT_DIR / "catalogue_repaired.xlsx", index=False)
print(f"rows: {len(df)}; original detail columns disagreed with the product name in {(df['cols_match_name'] == 'False').sum()} rows")