"""Intent detection and evidence retrieval inside ONE drug's leaflet (plan 1.10, 1.11)."""
import json
import re
from collections import defaultdict
from functools import lru_cache

from config import CHUNKS_FILE
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
    ("price", r"qiymet|nece manat|\bazn\b|bahadir|ucuzdur"),
    ("maker", r"istehsal|hansi sirket|hansi olke"),
    ("package", r"qablasdirma|qutuda|nece (eded|tablet|ampul|kapsul) (var|olur)"),
    ("contraindications", r"eks gosteris|olarmi|olar mi|olmaz|bilmez|bilmerem|bilerem|bilermi|qadagan|etmemeli|isletmek|icmek"),
    ("side_effects", r"yan tesir|elave tesir|yan effekt|zerer|reaksiya|a?l+erg"),
    ("dosage", r"\bdoza|nece (qebul|ic|istifade|vur)|gunde nece|nece defe|qebul qaydasi|istifade qaydasi|yemekden (evvel|sonra)|ac qarina|tox qarina"),
    ("indications", r"ne ucun|hansi (xestelik|hallarda|hal)\w*|ise yarayir|istifadesine gosteris|ne vaxt istifade"),
    ("interactions", r"qarsiliqli|birlikde|eyni vaxtda|ile birge|alkoqol|alkohol|spirtli|basqa derman"),
    ("pregnancy", r"hamile|emzir|laktasiya"),
    ("pediatric", r"usaq|korpe|pediatr"),
    ("overdose", r"doza hedd|artiq (icmis|qebul)|heddinden artiq|zeherlen|overdoz"),
    ("storage", r"saxla|soyuducu"),
    ("shelf_life", r"yararli|son istifade|bitme tarix|vaxti kecib"),
    ("composition", r"terkib|tesiredici madde|nelerden ibaret|icinde ne"),
    ("driving", r"masin sur|surucu|neqliyyat"),
    ("special", r"xususi gosteris|ehtiyat tedbir"),
    ("dispensing", r"resept|receptsiz|resepsiz|aptek"),
    ("explain", r"ne demek|hansilardir|izah|aciqla|menasi|sade dille|basa dus"),
]
CATALOGUE_INTENTS = {"price", "maker", "package"}

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


def retrieve(chunks, question, brand_key, intents, max_chunks=4, max_chars=3200):
    """Best chunks of ONE drug: the sections the question is about + word overlap with the question."""
    wanted = intents & CATEGORIES
    drop = set(brand_key.split()) | STOP
    q_stems = {w[:5] for w in search_key(question).split() if len(w) > 3 and w not in drop and not w.isdigit()}
    scored = []
    for c in chunks:
        stems = {w[:5] for w in search_key(c["text"]).split() if len(w) > 3}
        overlap = len(q_stems & stems)
        if c["category"] in wanted:
            scored.append((100 + overlap, c))
        elif overlap >= (2 if wanted else 1):
            scored.append((overlap, c))
    scored.sort(key=lambda x: (-x[0], x[1]["chunk_index"]))
    picked = sorted((c for _, c in scored[:max_chunks]), key=lambda c: c["chunk_index"])
    out, used = [], 0
    for c in picked:
        room = max_chars - used
        if room < 200:
            break
        out.append({**c, "text": c["text"][:room]})
        used += len(out[-1]["text"])
    return out


def unknown_only(evidence, intents):
    """True when the wanted section only says 'Məlum deyil.' (not known)."""
    w = [c for c in evidence if c["category"] in intents]
    return bool(w) and all(search_key(c["text"]) == "melum deyil" for c in w)