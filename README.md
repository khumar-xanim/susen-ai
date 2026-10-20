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