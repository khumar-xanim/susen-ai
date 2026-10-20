from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCEL = ROOT /"data"/ "reseptim_dermanlar.xlsx"
PDF_DIR = ROOT / "pdf_files"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

COL_ID = "ID"
COL_NAME = "Dərmanın adı"
COL_PRICE = "Qiymət (AZN)"
COL_FORM = "Farmasevtik forması"
COL_INTAKE = "Qəbul forması"
COL_COUNTRY = "Ölkə"
COL_PDF = "PDF_Y_N"

TEMPLATE_DIR = ROOT / "templates"
EXPECTED_COLUMNS = [
    "ID", "Sıra №", "Dərmanın adı", "Qiymət (AZN)", "Farmasevtik forması",
    "Ticarət qablaşdırması", "Miqdar", "Təsiredici maddənin adı", "Dozası",
    "İstehsalçı şirkət", "Ölkə", "Qəbul forması", "Qablaşdırma miqdarı", "PDF_Y_N",
]