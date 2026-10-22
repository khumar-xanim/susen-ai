"""Photo helpers (plan 1.7): the drug name is the BIGGEST text on a box or leaflet; ingredients are often in brackets."""
import re

from retrieval import _is_heading_line

INN = re.compile(r"\b[a-z]{4,}(?:mycin|micin|cillin|floxacin|azole|prazole|pril|sartan|statin|dipine|olol|sone|solone|"
                 r"formin|gliptin|glutide|caine|profen|fenac|zepam|oxetine|triptan|tinib|mab)\b", re.I)


def clean_word(w):
    return re.sub(r"[^A-Za-zƏəĞğIıİiÖöŞşÜüÇç0-9]", "", w)


def name_candidates(words, top=6):
    ws = []
    for w in words:
        c = clean_word(w["text"])
        if sum(ch.isalpha() for ch in c) >= 4:
            ws.append({**w, "clean": c})
    if not ws:
        return []
    hmax = max(w["h"] for w in ws)
    out = []
    for w in sorted((w for w in ws if w["h"] >= 0.55 * hmax), key=lambda w: (-w["h"], -w["conf"])):
        if w["clean"].lower() not in {o.lower() for o in out}:
            out.append(w["clean"])
    return out[:top]


def active_candidates(text):
    inside = " ".join(re.findall(r"\(([^)]{4,80})\)", text))
    found = [clean_word(t.strip().split(" ")[0]).lower() for t in re.split(r"[,;+/&]|\band\b|\bvə\b", inside) if t.strip()]
    found += [m.group(0).lower() for m in INN.finditer(text)]
    return [a for a in dict.fromkeys(found) if len(a) >= 5]


def leaflet_like(text):
    lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) < 90]
    return len(text) > 700 and sum(_is_heading_line(l) for l in lines) >= 2