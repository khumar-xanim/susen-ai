"""Answer checks shared by the chatbot and the evaluation (plan 1.13, 1.17)."""
import re

from normalize import search_key
from retrieval import stem

FOREIGN = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u3040-\u30FF\u4E00-\u9FFF]")   # Cyrillic, Arabic, Japanese, Chinese
NUM = re.compile(r"\d+(?:[.,]\d+)?")
ABSTAIN = re.compile(r"melumat (tapmadim|yoxdur)")
_TEMPLATE = ("Təlimat vərəqəsinə görə bu dərmanı bu hallarda istifadə etmək olmaz göstərilib yazılıb əlavə təsirlər "
             "bunlardır tez-tez nadir hallarda həkim əczaçı qəbul qaydası doza belə")
COMMON = {stem(w) for w in search_key(_TEMPLATE).split() if len(w) > 3}


def _stems(text):
    return {stem(w) for w in search_key(text).split() if len(w) > 3 and not w.isdigit()}


def support_score(answer, evidence_text):
    """Share of answer sentences whose content words (>= 50 %) occur in the evidence. A proxy for faithfulness."""
    ev = _stems(evidence_text)
    sents = [s for s in re.split(r"(?<=[.!?;:])\s+", answer) if len(s.strip()) > 15]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        st = _stems(s) - COMMON
        if not st or len(st & ev) / len(st) >= 0.5:
            ok += 1
    return ok / len(sents)


def unsupported_numbers(answer, evidence_text):
    norm = lambda t: {n.replace(",", ".") for n in NUM.findall(t)}
    return sorted(norm(answer) - norm(evidence_text))


def check_answer(answer, evidence_text, min_support=0.5):
    problems = []
    if FOREIGN.search(answer):
        problems.append("foreign_script")
    if len(answer) > 1200:
        problems.append("too_long")
    score = support_score(answer, evidence_text)
    if score < min_support:
        problems.append(f"low_support:{score:.2f}")
    nums = unsupported_numbers(answer, evidence_text)
    if nums:
        problems.append("numbers:" + ",".join(nums))
    return problems, round(score, 2)


def is_abstention(text):
    return bool(ABSTAIN.search(search_key(text)))