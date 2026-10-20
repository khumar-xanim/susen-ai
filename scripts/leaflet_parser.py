"""Leaflet PDF parsing and chunking (plan 1.6 and 1.8).
Text-layer PDFs are parsed here; pages without usable text are flagged for OCR."""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

MIN_TEXT_CHARS = 30    # fewer characters on a page = no usable text layer
MAX_GARBLED = 0.05     # share of broken characters that marks a page as unreadable
EDGE = 0.12            # top/bottom 12% of a page = header/footer zone
MAX_CHARS = 1200       # target maximum chunk size
OVERLAP_CHARS = 200    # a last sentence up to this long is repeated in the next chunk
PREAMBLE = "Ümumi məlumat"

CHAR_FIXES = str.maketrans({"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi",
                            "\ufb04": "ffl", "\u00ad": "", "\u00a0": " ", "\u200b": ""})
BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)
BULLET = re.compile(r"^(•|·|-|–|—|\d{1,2}[.)])\s")
PAGE_NUM = re.compile(r"^(s[əe]hif[əe]\s*)?\d{1,3}(\s*/\s*\d{1,3})?$", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def clean(text):
    text = unicodedata.normalize("NFC", text.translate(CHAR_FIXES))
    return re.sub(r"[ \t]+", " ", text).strip()


def _is_bold(span):
    return bool(span["flags"] & 16) or bool(BOLD_FONT.search(span["font"]))


def _page_lines(page, pno):
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            real = [s for s in line["spans"] if s["text"].strip()]
            if not real:
                continue
            text = clean("".join(s["text"] for s in line["spans"]))
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append({"page": pno, "x0": x0, "y0": y0, "y1": y1,
                          "size": max(s["size"] for s in real),
                          "bold": all(_is_bold(s) for s in real), "text": text})
    return lines


def _group_rows(lines):
    """Join pieces that sit on the same visual row (e.g. 'Tərkibi:' label + its text)."""
    lines = sorted(lines, key=lambda l: ((l["y0"] + l["y1"]) / 2, l["x0"]))
    groups = []
    for l in lines:
        mid = (l["y0"] + l["y1"]) / 2
        if groups and abs(mid - groups[-1]["mid"]) <= 0.4 * (l["y1"] - l["y0"]):
            groups[-1]["parts"].append(l)
        else:
            groups.append({"mid": mid, "parts": [l]})
    rows = []
    for g in groups:
        p = sorted(g["parts"], key=lambda l: l["x0"])
        rows.append({"page": p[0]["page"], "y0": min(x["y0"] for x in p),
                     "y1": max(x["y1"] for x in p), "size": max(x["size"] for x in p),
                     "bold": all(x["bold"] for x in p), "text": " ".join(x["text"] for x in p)})
    return rows


def _key(text):
    return re.sub(r"\d+", "#", text.casefold())


def extract_rows(doc):
    """Return (rows of all readable pages without headers/footers, {page: status})."""
    pages, status = [], {}
    for pno, page in enumerate(doc, start=1):
        rows = _group_rows(_page_lines(page, pno))
        height = page.rect.height
        for r in rows:
            r["edge"] = r["y1"] < EDGE * height or r["y0"] > (1 - EDGE) * height
        text = "".join(r["text"] for r in rows)
        if len(text) < MIN_TEXT_CHARS:
            status[pno] = "scan" if page.get_images() else "blank"
        elif sum(1 for c in text if c == "\ufffd" or "\ue000" <= c <= "\uf8ff") / len(text) > MAX_GARBLED:
            status[pno] = "garbled"
        else:
            status[pno] = "text"
            pages.append(rows)

    # drop page numbers and lines repeated on most pages inside the header/footer zone
    seen = Counter()
    for rows in pages:
        for k in {_key(r["text"]) for r in rows if r["edge"]}:
            seen[k] += 1
    cutoff = max(3, 0.6 * len(pages))
    kept = []
    for rows in pages:
        for r in rows:
            repeated = len(pages) >= 3 and seen[_key(r["text"])] >= cutoff and len(r["text"]) < 120
            if r["edge"] and (PAGE_NUM.match(r["text"]) or repeated):
                continue
            kept.append(r)
    return kept, status


def is_heading(row):
    t = row["text"]
    return (row["bold"] and 2 <= len(t) <= 90 and len(t.split()) <= 12
            and re.search(r"[^\W\d_]", t) is not None)


def _join(texts):
    """Join wrapped lines; repair words hyphenated across a line break."""
    out = ""
    for t in texts:
        if not out:
            out = t
        elif re.search(r"[^\W\d_]-$", out) and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    return out


def build_sections(rows):
    """Split rows into sections at bold, whole-line headings. Returns (sections, headings)."""
    sections, headings = [], []
    cur = {"heading": PREAMBLE, "paras": []}
    para = None

    def flush():
        nonlocal para
        if para:
            cur["paras"].append((para["p0"], para["p1"], _join(para["texts"])))
            para = None

    for r in rows:
        if is_heading(r):
            flush()
            h = r["text"].rstrip(":").strip()
            headings.append(h)
            if cur["paras"]:
                sections.append(cur)
                cur = {"heading": h, "paras": []}
            elif cur["heading"] == PREAMBLE:
                cur["heading"] = h
            elif h[:1].islower():       # wrapped heading: second line continues the first
                cur["heading"] += " " + h
            else:                       # heading directly under a heading: keep both
                cur["heading"] += " / " + h
            continue
        starts_new = (para is None or BULLET.match(r["text"])
                      or (r["page"] == para["last"]["page"]
                          and r["y0"] - para["last"]["y1"] > 0.35 * r["size"]))
        if starts_new:
            flush()
            para = {"p0": r["page"], "p1": r["page"], "texts": [r["text"]], "last": r}
        else:
            para["texts"].append(r["text"])
            para["p1"] = r["page"]
            para["last"] = r
    flush()
    if cur["paras"]:
        sections.append(cur)
    return sections, headings


def _units(paras, max_chars):
    """Break paragraphs into pieces of at most max_chars, keeping page numbers."""
    for p0, p1, text in paras:
        if len(text) <= max_chars:
            yield p0, p1, text
            continue
        buf = ""
        for s in SENTENCE_END.split(text):
            while len(s) > max_chars:
                if buf:
                    yield p0, p1, buf
                    buf = ""
                cut = s.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                yield p0, p1, s[:cut]
                s = s[cut:].lstrip()
            if buf and len(buf) + 1 + len(s) > max_chars:
                yield p0, p1, buf
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            yield p0, p1, buf


def chunk_section(section, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    groups, cur, size = [], [], 0
    for u in _units(section["paras"], max_chars):
        if cur and size + len(u[2]) + 1 > max_chars:
            groups.append(cur)
            tail = cur[-1]
            cur = [tail] if len(tail[2]) <= overlap else []
            size = sum(len(x[2]) + 1 for x in cur)
        cur.append(u)
        size += len(u[2]) + 1
    if cur:
        groups.append(cur)
    return [{"page_start": min(u[0] for u in g), "page_end": max(u[1] for u in g),
             "text": "\n".join(u[2] for u in g)} for g in groups]


def parse_pdf_path(path_str):
    """Parse one PDF. Returns (log_row, chunks, headings). Never raises."""
    path = Path(path_str)
    log = {"id": path.stem, "file": path.name, "pages": 0, "text_pages": 0, "ocr_pages": "",
           "chars": 0, "n_headings": 0, "n_sections": 0, "n_chunks": 0, "status": "ok", "error": ""}
    chunks, headings = [], []
    try:
        with fitz.open(path) as doc:
            log["pages"] = len(doc)
            rows, status = extract_rows(doc)
        ocr_pages = [p for p, s in status.items() if s in ("scan", "garbled")]
        log["text_pages"] = sum(1 for s in status.values() if s == "text")
        log["ocr_pages"] = ",".join(map(str, ocr_pages))
        if not rows:
            log["status"] = "needs_ocr" if ocr_pages else "no_text"
        else:
            if ocr_pages:
                log["status"] = "partial_ocr"
            sections, headings = build_sections(rows)
            idx = 0
            for sec in sections:
                for ch in chunk_section(sec):
                    chunks.append({"chunk_id": f"{log['id']}#{idx:03d}", "medicine_id": log["id"],
                                   "section": sec["heading"], "page_start": ch["page_start"],
                                   "page_end": ch["page_end"], "chunk_index": idx,
                                   "text": ch["text"], "n_chars": len(ch["text"]),
                                   "source_file": path.name})
                    idx += 1
            log.update(chars=sum(c["n_chars"] for c in chunks), n_headings=len(headings),
                       n_sections=len(sections), n_chunks=len(chunks))
    except Exception as e:  # corrupt PDF etc.
        log["status"], log["error"] = "error", str(e)[:200]
    return log, chunks, headings