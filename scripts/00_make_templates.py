"""Create empty placeholders: an Excel with headers only and an empty pdf_files folder."""
import pandas as pd
from config import TEMPLATE_DIR, EXPECTED_COLUMNS

(TEMPLATE_DIR / "pdf_files").mkdir(parents=True, exist_ok=True)
# Git does not track empty folders, so a hidden .gitkeep file holds the folder in place.
(TEMPLATE_DIR / "pdf_files" / ".gitkeep").write_text("", encoding="utf-8")

pd.DataFrame(columns=EXPECTED_COLUMNS).to_excel(
    TEMPLATE_DIR / "reseptim_dermanlar.xlsx", index=False
)
print("Templates created in", TEMPLATE_DIR)