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
    ("side_effects", r"elave tesir|arzuolunmaz|zedelenme|meddra"),
    ("interactions", r"qarsiliqli tesir"),
    ("pregnancy", r"hamilelik|laktasiya"),
    ("special", r"xususi gosteris|ehtiyat|ehtiyyat|uyusmazlig|yaslilarda"),
    ("driving", r"neqliyyat"),
    ("pediatric", r"pediatr|usaqlarda|yeniyetme"),
    ("storage", r"saxlan|saxlam"),
    ("shelf_life", r"yararl"),
    ("composition", r"terkib"),
    ("description", r"tesvir"),
    ("dispensing", r"apekden|aptekden|apteklerden"),
    ("form", r"buraxilis forma"),
    ("manufacturer_info", r"istehsal|qeydiyyat|distribyuter|lisenziya|unvan|numayendelik|iradlari qebul|qablasdirici|marketinq icazesi|selahiyyetli"),
    ("pharm_group", r"farmakoterapevtik"),
    ("pharmacology", r"farmako|mikroorqanizm|aerob|anaerob|qrammusbet|qrammenfi"),
]
CATEGORIES = {name for name, _ in SECTION_PATTERNS}
INTENT_PATTERNS = [
    ("price", r"qiymet|neceyedir|nece manat|nece pul|\bazn\b|bahadir|ucuzdur|\bprice\b|\bcost\b"),
    ("maker", r"istehsal|hansi sirket|hansi olke|manufactur|which company|made by"),
    ("package", r"qablasdirma|qutuda|nece (eded|tablet|ampul|kapsul) (var|olur)"),
    ("availability", r"hansi aptek|harada (tap|sat|al)|stokda"),
    ("summary", r"xulase|qisa mezmun|qisa melumat|qisaca|ne haqqindadir|haqqinda melumat|barede melumat|summary|overview|tell me about"),
    ("contraindications", r"eks gosteris|olarmi|olar mi|olmaz|bilmez|bilmerem|bilerem|bilermi|qadagan|etmemeli|isletmek|icmek|contraindicat|who should not|cannot take"),
    ("side_effects", r"yan tesir|elave tesir|yan effekt|zerer|reaksiya|a?l+erg|side effect|adverse"),
    ("dosage", r"\bdoza|nece (qebul|ic|istifade|vur)|gunde nece|nece defe|qebul qaydasi|istifade qaydasi|yemekden (evvel|sonra)|ac qarina|tox qarina|dosage|how to take|how often|how much"),
    ("route", r"icirler|icilir|vurulur|vururlar|inyeksiya ile|derialti|ezelede|damara|damardan|hansi formada|hansi sekilde"),
    ("indications", r"ne ucun|hansi (xestelik|hallarda|hal)\w*|ise yarayir|istifadesine gosteris|ne vaxt istifade|used for|indication"),
    ("interactions", r"qarsiliqli|birlikde|eyni vaxtda|ile birge|alkoqol|alkohol|spirtli|basqa derman|interact|with alcohol"),
    ("pregnancy", r"hamile|emzir|laktasiya|pregnan|breast ?feed|lactation"),
    ("pediatric", r"usa[qg]|korpe|pediatr|\d+ ?(yasli|yasinda|ayliq)|children"),
    ("overdose", r"doza hedd|artiq (icmis|qebul)|heddinden artiq|zeherlen|overdoz"),
    ("storage", r"saxla|soyuducu|\bstorage\b|\bstore\b"),
    ("shelf_life", r"yararli|son istifade|bitme tarix|vaxti kecib|expir"),
    ("composition", r"terkib|tesiredici madde|nelerden ibaret|icinde ne|ingredient"),
    ("driving", r"masin sur|surucu|neqliyyat"),
    ("special", r"xususi gosteris|ehtiyat tedbir"),
    ("dispensing", r"resept|receptsiz|resepsiz|aptekden buraxil"),
    ("explain", r"ne demek|hansilardir|izah|aciqla|menasi|sade dille|basa dus|what does .* mean"),
]
SUMMARY_SECTIONS = ["pharm_group", "pharmacology", "indications", "contraindications", "dosage"]

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


EXPAND = {  # query expansion: words the user types -> words the leaflet uses
    "emzir": "ana sudu laktasiya", "sud": "ana sudu laktasiya", "gunes": "gunes isigi fotohessasliq ultrabenovseyi",
    "isiq": "gunes isigi fotohessasliq", "alkoqol": "alkoqol spirt", "alkohol": "alkoqol spirt",
    "dayandir": "dayandirilmasi kesilmesi", "yarimciq": "dayandirilmasi", "surucu": "neqliyyat masin mexanizm",
    "masin": "neqliyyat masin mexanizm", "hamile": "hamilelik", "usaq": "usaq korpe yas", "yasli": "yas yaslilar",
    "breast": "ana sudu laktasiya", "sunlight": "gunes isigi fotohessasliq", "pregnan": "hamilelik",
    "alcohol": "alkoqol spirt", "driv": "neqliyyat masin",
}


def _qstems(question, brand_key=""):
    drop = set(brand_key.split()) | STOP
    words = [w for w in search_key(question).split() if len(w) > 3 and w not in drop and not w.isdigit()]
    stems = {stem(w) for w in words}
    for w in words:
        for k, extra in EXPAND.items():
            if w.startswith(k):
                stems |= {stem(x) for x in extra.split()}
    return stems

DENSE_MIN = 0.50      # a chunk is also considered when its MEANING is close; tune with scripts\20_dense_check.py
DENSE_W = 40


def rank_chunks(chunks, question, brand_key, wanted, dense=None):
    """All useful chunks of ONE drug, best first: wanted sections + shared words + embedding similarity."""
    q_stems = _qstems(question, brand_key)
    thr = DENSE_MIN + (0.10 if wanted else 0.0)
    scored = []
    for c in chunks:
        overlap = len(q_stems & {stem(w) for w in search_key(c["text"]).split() if len(w) > 3})
        d = (dense or {}).get(c["chunk_index"], 0.0)
        bonus = DENSE_W * max(d - 0.30, 0.0)
        if c["category"] in wanted:
            scored.append((100 + overlap + bonus, c))
        elif overlap >= (2 if wanted else 1) or d >= thr:
            scored.append((overlap + bonus, c))
    scored.sort(key=lambda x: (-x[0], x[1]["chunk_index"]))
    return scored


def retrieve(chunks, question, brand_key, wanted, max_chunks=3, max_chars=2800, dense=None):
    ranked = rank_chunks(chunks, question, brand_key, wanted, dense)
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


def excerpt(evidence, wanted, question="", limit=450):
    """The sentences of the original leaflet that match the question best. Returns (text, chunk)."""
    q = _qstems(question)

    def score(text):
        return len(q & {stem(w) for w in search_key(text).split() if len(w) > 3})

    pool = [c for c in evidence if c["category"] in wanted] or evidence
    c = max(pool, key=lambda c: (score(c["text"]), -c["chunk_index"]))
    sents = re.split(r"(?<=[.!?;])\s+", c["text"])
    start = max(range(len(sents)), key=lambda i: (score(sents[i]), -i))
    out = ""
    for s in sents[start:]:
        if out and len(out) + len(s) > limit:
            break
        out += (" " if out else "") + s
    return (out if len(out) <= limit + 150 else out[:limit].rsplit(" ", 1)[0] + " …"), c

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