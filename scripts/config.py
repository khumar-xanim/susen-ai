from pathlib import Path
import os
import hashlib

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

CHUNKS_FILE = OUT_DIR / "chunks.jsonl"
LLM_PROVIDER = os.getenv("SUSEN_PROVIDER", "ollama")
LLM_MODEL = os.getenv("SUSEN_MODEL", "qwen3.5:4b")
LLM_OPTIONS = {"temperature": 0.0, "top_p": 1.0, "num_ctx": 4096, "num_predict": 600, "seed": 42}
ADMIN = os.getenv("SUSEN_ADMIN") == "1"      # admin view: file names, model setting, error details

TESSDATA_DIR = ROOT / "tessdata"
EVAL_DIR = OUT_DIR / "eval"
EVAL_DIR.mkdir(exist_ok=True)


def split_of(med_id):
    """Deterministic dev/test split BY DRUG (20 % test): tune on dev, report on test."""
    return "test" if int(hashlib.md5(med_id.encode()).hexdigest(), 16) % 5 == 0 else "dev"
