# SUSEN AI - Azerbaijani Drug Information Assistant

Local chatbot that answers questions about medicines using a structured
catalogue (Excel, prices in AZN) and leaflet PDFs. Identity, price, strength
and packaging come from the catalogue, never from the LLM.

## Project layout

    reseptim_dermanlar.xlsx   catalogue (4210 rows, price column: Qiymət (AZN))
    pdf_files/                leaflet PDFs (not in Git)
    scripts/                  numbered step scripts
    reports/                  text reports from each step
    outputs/                  generated CSVs (not in Git)

## Setup (Windows, Python 3.12)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Progress log

### Step 0 - Scaffold
Folders, requirements.in / requirements.txt (frozen versions), .gitignore, README, GitHub repo.
### Step 1 - Catalogue audit (plan 1.1)
Script: `scripts/01_audit_catalogue.py` (paths and column names in `scripts/config.py`).
Full output: `reports/01_audit_report.txt`.

Findings:
- 4210 rows, 14 columns, no duplicate IDs or product strings.
- Empty cells: Miqdar 45, Dozası 8, Qəbul forması 24, Qablaşdırma miqdarı 24.
- Prices in AZN: min 0.10, median 11.04, max 47033 (outlier to verify in Step 2).
- Form typos to fix: "KAPSUL", "İnfuzya üçün məhlul".
- Qəbul forması uses both "Ədəd" and "Tablet" for tablets (unify in Step 4).
- PDF_Y_N: 3478 rows flagged 1, 732 flagged 0.

Run: `python scripts\01_audit_catalogue.py`
### Step 1b - Data is not published
The catalogue and leaflets are not in this repository. `templates/` contains an empty
`reseptim_dermanlar.xlsx` (column headers only) and an empty `pdf_files/` folder.
To run the project, copy them to the project root and fill them with your own data:

    Copy-Item templates\reseptim_dermanlar.xlsx .
    Copy-Item templates\pdf_files . -Recurse

### Design decisions
- Prices are the official prices from the source site and are never modified.
  Suspicious values are only flagged (`price_review`) and reported to the authorities.
- Some leaflets are missing on the source site (empty HTML pages instead of PDFs);
  `PDF_Y_N = 0` marks them. If a leaflet or a catalogue field is missing, the chatbot
  must say it does not have that information instead of guessing.
- Leaflets will be chunked by bold headings; scanned leaflets go through OCR.
### Step 2 - Clean catalogue (plan 1.1)
Script: `scripts/02_clean_catalogue.py`. Output (not in Git): `outputs/catalogue_clean.csv`.
New columns, originals untouched:
- `form_clean`: fixed typos ("KAPSUL", "İnfuzya üçün məhlul")
- `price_azn`: numeric price (official value, unchanged); `price_review`: True if > 1000 AZN
- `has_leaflet`: from PDF_Y_N
- `missing_fields`: which catalogue fields are empty in that row

Run: `python scripts\02_clean_catalogue.py`