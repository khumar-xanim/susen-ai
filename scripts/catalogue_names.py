"""Compare the table columns with the product string 'BRAND, FORM, PACK, №N (ACTIVE DOSE, MAKER, COUNTRY)'."""
import re

import pandas as pd

from normalize import search_key

NAME_RE = re.compile(r"^(?P<brand>[^,]+), (?P<form>[^,]+), (?P<pack>[^()]+?)(?:, (?P<qty>№\S+))? \((?P<inner>.+)\)\s*$")


def parse_name(name):
    m = NAME_RE.match(str(name).strip())
    if not m:
        return None
    head, _, country = m["inner"].rpartition(", ")
    active_dose, _, maker = head.partition(", ")
    return {"form": m["form"].strip(), "pack": m["pack"].strip(), "qty": (m["qty"] or "").strip(),
            "active_dose": active_dose.strip(), "maker": maker.strip(), "country": country.strip()}


def _k(x):
    return "" if pd.isna(x) else search_key(x)


def reconcile(name, form, active, dose, maker, country, pack, qty):
    """Clean values. If the columns disagree with the product string, the string wins."""
    cols = {"form_clean": form, "active_clean": active, "dose_clean": dose, "maker_clean": maker,
            "country_clean": country, "pack_clean": pack, "qty_clean": qty}
    p = parse_name(name)
    if p is None:
        return {**cols, "cols_match_name": True, "mismatch_fields": "unparsed"}
    key = search_key(name)
    bad = []
    if _k(form) != search_key(p["form"]):
        bad.append("form")
    if _k(active) not in key:
        bad.append("active")
    if _k(dose) not in key:
        bad.append("dose")
    if _k(maker) not in key:
        bad.append("maker")
    if p["country"] and _k(country) != search_key(p["country"]):
        bad.append("country")
    if not bad:
        return {**cols, "cols_match_name": True, "mismatch_fields": ""}
    return {"form_clean": p["form"], "active_clean": p["active_dose"], "dose_clean": p["active_dose"],
            "maker_clean": p["maker"], "country_clean": p["country"], "pack_clean": p["pack"],
            "qty_clean": p["qty"], "cols_match_name": False, "mismatch_fields": ",".join(bad)}