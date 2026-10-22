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

### Step 3 - PDF extraction and chunking (plan 1.6, 1.8)
Code: `scripts/leaflet_parser.py`, `scripts/03_parse_pdfs.py` (parallel), `scripts/show_chunks.py`.
Outputs (not in Git): `outputs/chunks.jsonl`, `pdf_extraction_log.csv`, `heading_vocabulary.csv`.

- Text layer read with PyMuPDF; headings = whole-line bold text; one chunk per section,
  long sections split to ~1200 characters with sentence overlap.
- Each chunk keeps medicine_id, section, page range, source file, extraction date.
- Pages without a usable text layer are listed in `ocr_pages` for Step 4.

- 3479 PDFs parsed in ~1 min, ~108,900 chunks (mean ~500 chars), 3477 fully OK.
- Headings = whole-line bold text; wrapped headings are merged; one chunk per section,
  long sections split with sentence overlap. Chunks keep medicine_id, section, pages, source file, date.
- Scans: only BENDATOP_04f237 (no text) and QRAMMIDIN_NEO_ANESTETIKL_67467f (partial) need OCR - done at the end.
- Heading spelling variants exist (e.g. "Əks göstərişləri"); mapped to canonical sections in the retrieval step.


Run: `python scripts\03_parse_pdfs.py` then `python scripts\show_chunks.py <ID prefix>`

### Step 4 - Normaliser and product finder (plan 1.2, 1.4)
`scripts/normalize.py`: `search_key()` folds İ/ı/I/i, ə, ş, ç, ğ, ö, ü and symbols to a plain lowercase
key (original text is never overwritten). `scripts/product_matcher.py`: finds the product in a user
message with RapidFuzz over 1-4 word phrases; returns found / ambiguous (candidate list) / not_found;
a number in the message ("abizol 10") narrows by strength. Tests: `python -m pytest`.

Try: `python scripts\04_find_product.py "exinaseya komp dermani almisam"`

### Step 5-6 - Retrieval, safety rules and chatbot (plan 1.10, 1.11, 1.13, 1.14)
`scripts/rules.py` (deterministic safety rules before the LLM), `scripts/retrieval.py` (intent detection,
evidence from ONE drug's leaflet), `scripts/chatbot.py` (pipeline), `scripts/llm.py` (only file that talks
to a model: local Ollama, switch with `SUSEN_MODEL`), `scripts/05_chat.py` (terminal chat),
`scripts/06_demo.py` (demo questions on several models -> `outputs/demo_answers.md`).

- Answers in Azerbaijani with sources (medicine ID, section, page); disclaimer and dose/pregnancy warnings added by code.
- Price/manufacturer/package come from the catalogue. No leaflet or "Məlum deyil" -> "no information".
- Refused before the LLM: prompt injection, dose changes, substitution advice, diagnosis; emergencies get 103/112.
- Models tested: qwen3.5:4b and gemma3:4b (`ollama pull` both). Default: qwen3.5:4b.

First demo results: all safety rules and catalogue answers worked. Open problems: grammar/suffix errors in
free-written Azerbaijani, answers that go beyond the leaflet, wrong "no information" replies, retrieval misses
(q->ğ alternation: uşaq/uşağa), slow first model load. Possible data issue: ATROPIN_SULFAT catalogue form
(tablet) vs leaflet (1 ml solution). Fixes in Step 7.

Setup: `ollama pull qwen3.5:4b`. Run: `python scripts\05_chat.py`




### Step 7 - Leaflet-first data, pharmacist persona, checks (plan 1.1, 1.13, 1.14)
- `llm.py`: model warm-up and 30 min keep-alive; `chatbot.py`: stricter prompt (reuse leaflet wording, no
  greeting, no claims beyond the leaflet, one Azerbaijani example), 
- Policy: leaflet first; catalogue only for price and as fallback. If the table columns disagree with the product
  string (e.g. ATROPIN_SULFAT: columns said tablet), the string wins (`catalogue_names.py`, `*_clean` columns,
  `outputs/name_column_mismatch.csv`); prices of such rows carry a caution.
- `08_pdf_identity_check.py` flags PDFs whose first page does not mention the linked drug.
- `retrieval.py`: q/ğ stem fix (uşaq/uşağa), route questions (içirlər/inyeksiya), age questions, `neceyedir`;
  `07_section_coverage.py` lists unmapped headings.
- `chatbot.py`: warm pharmacist persona ("not a doctor"), strict Azerbaijani prompt with examples, answers only
  rephrase the leaflet and always show the leaflet file + page + section; `quality.py` checks every answer
  (foreign script, unsupported sentences/numbers), retries once, falls back to the original leaflet text.
- Deterministic (no LLM): price, manufacturer/packaging (leaflet first), summaries, safety rules (`rules.py`). drug-recommendation requests are refused; 
- `06_demo.py --file scripts/manual_questions.txt` -> `outputs/demo_answers.md` for manual review.
### Step 8 - Chat window (plan 1.16)
`streamlit run scripts\app.py`: chat bubbles, model + temperature selector, candidate buttons, source list with
"show leaflet page" (renders the PDF page), 👍/👎 feedback (`outputs/feedback.jsonl`), photo upload (Step 10).

### Step 9- OCR, photos and evaluation (plan 1.7, 1.17)
- `ocr.py` (deskew + adaptive threshold + Tesseract `aze`, needs `tessdata/aze.traineddata`, not in Git),
  `09_ocr_scans.py` adds OCR chunks for pages without text layer; headings of scans/photos are matched against
  `heading_vocabulary.csv`. Photo mode in the chat window: answers come from the user's photo, source "Sizin şəkil".
- Evaluation, dev/test split BY DRUG (`config.split_of`): `10_eval_make.py` (questions), `11_eval_run.py`
  (product identification, Recall@1/3/4, MRR vs keyword baseline, abstention, raw/final hallucination proxies,
  temperature sweep, direct + photo-embedded injection), `12_eval_score.py` (manual labels: correctness,
  faithfulness, hallucination), `13_eval_ocr.py` (simulated phone photos: CER, drug identification).
- Tune on dev only; report test once.
### Step 10- Fixes and hardening
- Data finding: table columns disagree with the product string in 4117/4210 rows (`14_alignment_audit.py`);
  form/composition/maker/country/pack come from the product string; prices carry a caution until verified.
- `normalize.py`: Cyrillic look-alike letters in leaflet headings are folded; section patterns extended;
  `leaflet_parser.is_heading` no longer takes sentences as headings. OCR: `TESSDATA_PREFIX` instead of quoted flag.
- Security: temperature fixed at 0 (seed 42); users never see file names/IDs/model/errors (public labels
  "<drug> təlimat vərəqəsi, səh. N"); admin view via `SUSEN_ADMIN=1`; page viewer only opens bot-selected files;
  session question limit and image size limit; model-server crash retry.
- `kb.py` (reader for the knowledge base, Step 11b), `08_pdf_identity_check.py` more tolerant.
### Step 11b - Knowledge base (plan 1.9, 1.10)
Trusted background texts for "what does this term mean / pharmacy practice" questions, retrieved (not memorised):
AMU lecture slides/DOCX (local, not in Git, permission needed for public use) + MedlinePlus health topics
(English; credit: "Source: MedlinePlus, National Library of Medicine"). `15_kb_build.py` (ingest + bge-m3
embeddings via Ollama), `kb.py`, `16_kb_search.py` (score check). Answers use ONLY retrieved passages, show the
source excerpt, and say "not found in reliable sources" when nothing is close enough (`MIN_SCORE`).
### Step 12 - Photos, natural answers, safety triage
- Photo: the biggest text and bracketed ingredients are read (`ocr.read_words`, `photo.py`), punctuation cleaned,
  matched to the catalogue by spelling skeleton (TobraDex ~ TOBRADEKS); several matches -> the user chooses.
- Answers are written by the model (no examples, no fixed openers/closers, no repeated disclaimer); up to 3
  generation attempts, never an apology/copy fallback; one short source line; summary = 5 leaflet titles.
- `rules.py`: triage before product matching (overdose, self-injection, harm, misuse, sedatives...). REPLY TEXTS
  NEED REVIEW BY A DOCTOR/PHARMACIST. `tests/test_triage.py` encodes the manual review questions.
- `18_consistency.py` measures how often a question is answered across repeated runs.