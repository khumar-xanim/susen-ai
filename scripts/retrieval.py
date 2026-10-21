"""Intent detection and evidence retrieval inside ONE drug's leaflet (plan 1.10, 1.11)."""
import json
import re
from collections import defaultdict
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from config import CHUNKS_FILE, OUT_DIR
from normalize import search_key

# Patterns work on search_key() text (lowercase, ə->e ı->i ş->s ç->c ğ->g ö->o ü->u).
SECTION_PATTERNS = [
    ("overdose", r"doza hedd"),
    ("contraindications", r"eks gosteris"),
    ("indications", r"istifadesine gosteris"),
    ("dosage", r"istifade qaydasi|dozasi"),
    ("side_effects", r"elave tesir|arzuolunmaz"),
    ("interactions", r"qarsiliqli tesir"),
    ("pregnancy", r"hamilelik|laktasiya"),
    ("special", r"xususi gosteris|ehtiyat|ehtiyyat"),
    ("driving", r"neqliyyat"),
    ("pediatric", r"pediatr"),
    ("storage", r"saxlan|saxlam"),
    ("shelf_life", r"yararliliq"),
    ("composition", r"terkib"),
    ("description", r"tesvir"),
    ("dispensing", r"apekden|aptekden"),
    ("form", r"buraxilis forma"),
    ("manufacturer_info", r"istehsal|qeydiyyat|distribyuter|lisenziya|unvan"),
    ("pharmacology", r"farmako"),
]
CATEGORIES = {name for name, _ in SECTION_PATTERNS}

INTENT_PATTERNS = [
    ("price", r"qiymet|neceyedir|nece manat|nece pul|\bazn\b|bahadir|ucuzdur"),
    ("maker", r"istehsal|hansi sirket|hansi olke"),
    ("package", r"qablasdirma|qutuda|nece (eded|tablet|ampul|kapsul) (var|olur)"),
    ("availability", r"hansi aptek|harada (tap|sat|al)|stokda"),
    ("summary", r"xulase|qisa mezmun|qisaca|ne haqqindadir|haqqinda melumat|barede melumat"),
    ("contraindications", r"eks gosteris|olarmi|olar mi|olmaz|bilmez|bilmerem|bilerem|bilermi|qadagan|etmemeli|isletmek|icmek"),
    ("side_effects", r"yan tesir|elave tesir|yan effekt|zerer|reaksiya|a?l+erg"),
    ("dosage", r"\bdoza|nece (qebul|ic|istifade|vur)|gunde nece|nece defe|qebul qaydasi|istifade qaydasi|yemekden (evvel|sonra)|ac qarina|tox qarina"),
    ("route", r"icirler|icilir|vurulur|vururlar|inyeksiya ile|derialti|ezelede|damara|damardan|hansi formada|hansi sekilde"),
    ("indications", r"ne ucun|hansi (xestelik|hallarda|hal)\w*|ise yarayir|istifadesine gosteris|ne vaxt istifade"),
    ("interactions", r"qarsiliqli|birlikde|eyni vaxtda|ile birge|alkoqol|alkohol|spirtli|basqa derman"),
    ("pregnancy", r"hamile|emzir|laktasiya"),
    ("pediatric", r"usa[qg]|korpe|pediatr|\d+ ?(yasli|yasinda|ayliq)"),
    ("overdose", r"doza hedd|artiq (icmis|qebul)|heddinden artiq|zeherlen|overdoz"),
    ("storage", r"saxla|soyuducu"),
    ("shelf_life", r"yararli|son istifade|bitme tarix|vaxti kecib"),
    ("composition", r"terkib|tesiredici madde|nelerden ibaret|icinde ne"),
    ("driving", r"masin sur|surucu|neqliyyat"),
    ("special", r"xususi gosteris|ehtiyat tedbir"),
    ("dispensing", r"resept|receptsiz|resepsiz|aptekden buraxil"),
    ("explain", r"ne demek|hansilardir|izah|aciqla|menasi|sade dille|basa dus"),
]
SUMMARY_SECTIONS = ["indications", "contraindications", "dosage", "side_effects", "pregnancy", "storage"]
EXTRA_SECTIONS = {"pediatric": {"dosage", "contraindications"}, "route": {"dosage", "form"},
                  "maker": {"manufacturer_info"}, "package": {"form"}, "summary": set(SUMMARY_SECTIONS)}

STOP = {"ve", "ile", "nedir", "derman", "dermani", "dermanin", "olan", "olanda", "ucun", "hansi", "necedir",
        "edir", "olar", "olarmi", "olmaz", "mene", "menim", "almisam", "isletmek", "demekdir", "hansilardir",
        "nece", "uygundur", "bilerem"}


@lru_cache(maxsize=None)
def category_of(section):
    key = search_key(section)
    for name, pat in SECTION_PATTERNS:
        if re.search(pat, key):
            return name
    return "other"


def load_chunks():
    by_id = defaultdict(list)
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_id[c["medicine_id"]].append({
                "section": c["section"], "page_start": c["page_start"], "page_end": c["page_end"],
                "text": c["text"], "chunk_index": c["chunk_index"], "category": category_of(c["section"])})
    return by_id


def detect_intents(message):
    key = search_key(message)
    return {name for name, pat in INTENT_PATTERNS if re.search(pat, key)}


def wanted_categories(intents):
    wanted = set()
    for i in intents:
        if i in CATEGORIES:
            wanted.add(i)
        wanted |= EXTRA_SECTIONS.get(i, set())
    return wanted


def stem(w):
    return w[:5].replace("g", "q")      # crude stem; g->q because of the q/ğ alternation (uşaq -> uşağa)


def rank_chunks(chunks, question, brand_key, wanted):
    """All useful chunks of ONE drug, best first: wanted sections + word overlap with the question."""
    drop = set(brand_key.split()) | STOP
    q_stems = {stem(w) for w in search_key(question).split() if len(w) > 3 and w not in drop and not w.isdigit()}
    scored = []
    for c in chunks:
        overlap = len(q_stems & {stem(w) for w in search_key(c["text"]).split() if len(w) > 3})
        if c["category"] in wanted:
            scored.append((100 + overlap, c))
        elif overlap >= (2 if wanted else 1):
            scored.append((overlap, c))
    scored.sort(key=lambda x: (-x[0], x[1]["chunk_index"]))
    return scored


def retrieve(chunks, question, brand_key, wanted, max_chunks=4, max_chars=3200):
    ranked = rank_chunks(chunks, question, brand_key, wanted)
    picked = sorted((c for _, c in ranked[:max_chunks]), key=lambda c: c["chunk_index"])
    out, used = [], 0
    for c in picked:
        room = max_chars - used
        if room < 200:
            break
        out.append({**c, "text": c["text"][:room]})
        used += len(out[-1]["text"])
    return out


def unknown_only(evidence, wanted):
    w = [c for c in evidence if c["category"] in wanted]
    return bool(w) and all(search_key(c["text"]) == "melum deyil" for c in w)


def excerpt(evidence, wanted, limit=500):
    """Original leaflet text (first matching chunk), cut at a sentence end."""
    pick = [c for c in evidence if c["category"] in wanted] or evidence
    text = pick[0]["text"]
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:end + 1] if end > 150 else cut.rsplit(" ", 1)[0] + " …"


# ---- OCR text -> chunks (scans and photos have no bold, so headings are matched against known headings) ----
@lru_cache(maxsize=None)
def _heading_keys():
    path = OUT_DIR / "heading_vocabulary.csv"
    if not path.exists():
        return ()
    df = pd.read_csv(path)
    return tuple({search_key(h) for h, n in zip(df["heading"], df["count"]) if n >= 30})


def _is_heading_line(line):
    k = search_key(line)
    if not k or len(k.split()) > 14:
        return False
    keys = _heading_keys()
    if keys:
        return process.extractOne(k, keys, scorer=fuzz.ratio, score_cutoff=88) is not None
    return len(k.split()) <= 6 and category_of(line) != "other"


def split_by_headings(text, page=1, start_index=0):
    from leaflet_parser import chunk_section
    sections, cur, buf = [], {"heading": "Ümumi məlumat", "paras": []}, []

    def flush():
        if buf:
            cur["paras"].append((page, page, " ".join(buf)))
            buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif _is_heading_line(line):
            flush()
            if cur["paras"]:
                sections.append(cur)
            cur = {"heading": line.rstrip(":"), "paras": []}
        else:
            buf.append(line)
    flush()
    if cur["paras"]:
        sections.append(cur)
    out = []
    for s in sections:
        for ch in chunk_section(s):
            out.append({"section": s["heading"], "page_start": ch["page_start"], "page_end": ch["page_end"],
                        "text": ch["text"], "chunk_index": start_index + len(out), "category": category_of(s["heading"])})
    return out