"""Evaluation (plan 1.17). Usage:
  python scripts\\11_eval_run.py --split dev                       (fast, no LLM: retrieval, abstention, rules)
  python scripts\\11_eval_run.py --split dev --llm --n-llm 30 --temperatures 0.0 0.3 0.7 [--model qwen3.5:4b]"""
import argparse
import json
import random

import pandas as pd

from chatbot import Chatbot, DETERMINISTIC
from config import EVAL_DIR, REPORT_DIR
from quality import check_answer
from retrieval import detect_intents, rank_chunks, wanted_categories
from dense import LEAFLET_DENSE
from kb import embed
ABSTAIN = {"no_product", "unknown_drug", "off_topic", "health_general", "no_leaflet", "no_evidence", "unknown_in_leaflet", "abstained", "kb_none"}

lines = []

def log(t=""):
    print(t)
    lines.append(str(t))


def load(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--llm", action="store_true")
ap.add_argument("--n-llm", type=int, default=30)
ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1])
ap.add_argument("--model")
args = ap.parse_args()

bot = Chatbot(model=args.model)
bot.no_llm = not args.llm
qs = load(EVAL_DIR / f"questions_{args.split}.jsonl")
by_type = {}
for q in qs:
    by_type.setdefault(q["type"], []).append(q)
log(f"=== {args.split} | model {bot.model} | " + ", ".join(f"{k}: {len(v)}" for k, v in by_type.items()))

# ---------- retrieval ----------
rows = []
for q in by_type.get("answerable", []):
    res = bot.index.find(q["question"])
    ids = {c["id"] for c in res["candidates"]}
    brand_key = bot.index.row(q["gold_id"])["brand_key"]
    wanted = wanted_categories(detect_intents(q["question"]) - DETERMINISTIC)
    row = {"product_found": q["gold_id"] in ids and res["status"] in ("found", "ambiguous"),
           "intent_detected": q["gold_category"] in wanted}
    variants = [("system", wanted, None), ("baseline", set(), None)]
    if LEAFLET_DENSE.ok:
        dn = LEAFLET_DENSE.scores(q["gold_id"], embed([q["question"]], cpu=True)[0])
        variants += [("system_dense", wanted, dn), ("dense_only", set(), dn)]
    for name, w, dn_ in variants:
        ranked = rank_chunks(bot.chunks[q["gold_id"]], q["question"], brand_key, w, dense=dn_)
        pos = next((i for i, (_, c) in enumerate(ranked, 1) if c["category"] == q["gold_category"]), None)
        for k in (1, 3, 4):
            row[f"{name}_recall@{k}"] = bool(pos and pos <= k)
        row[f"{name}_mrr"] = 1 / pos if pos else 0.0
    rows.append(row)
if rows:
    ret = pd.DataFrame(rows)
    log("\nRETRIEVAL on answerable questions (system = intents + keywords, baseline = keywords only)")
    log(ret.mean().round(3).to_string())

# ---------- abstention ----------
log("\nABSTENTION: share of questions that were correctly refused / answered with 'no information'")
for typ in ("unknown_section", "no_leaflet", "fake_drug", "offtopic"):
    ok, misses, skipped = [], [], 0
    for q in by_type.get(typ, []):
        bot.reset()
        st = bot.ask(q["question"])["status"]
        if st == "llm_skipped":
            skipped += 1                 # it reached the model: judge it with --llm
            continue
        good = st in ABSTAIN or st.startswith("rule:")
        ok.append(good)
        if not good:
            misses.append(f"{q['question']}  ->  {st}")
    if ok or skipped:
        log(f"  {typ:<16} {sum(ok)}/{len(ok)} refused correctly" + (f"  ({skipped} reached the model)" if skipped else ""))
        for m in misses[:5]:
            log("      MISS: " + m)

# ---------- LLM answers, temperature sweep ----------
def run_llm(temp, n):
    bot.options = {"temperature": temp}
    sample = random.Random(1).sample(by_type.get("answerable", []), min(n, len(by_type.get("answerable", []))))
    recs = []
    for q in sample:
        bot.reset()
        out = bot.ask(q["question"])
        text, ev = out.get("llm_text", ""), out.get("evidence_text", "")
        problems, support = check_answer(text, ev) if text else (["no_llm_text"], 0.0)
        recs.append({"qid": q["qid"], "type": q["type"], "temp": temp, "question": q["question"], "gold_id": q["gold_id"],
                     "gold_category": q["gold_category"], "status": out["status"], "reply": out["reply"], "llm_text": text,
                     "evidence_text": ev, "proxy_support": support, "proxy_problems": ";".join(problems),
                     "attempt1_problems": ";".join(out.get("attempt1_problems") or []), "seconds": out["seconds"],
                     "label_correct": "", "label_faithful": "", "label_halluc": "", "notes": ""})
    return pd.DataFrame(recs)


if args.llm:
    table = {}
    for t in args.temperatures:
        df = run_llm(t, args.n_llm)
        df.to_csv(EVAL_DIR / f"answers_{args.split}_{bot.model.replace(':', '-')}_T{t}.csv", index=False, encoding="utf-8-sig")
        has = df[df["llm_text"] != ""]
        answered = df[df["status"] == "answered"]
        table[f"T={t}"] = {
            "n": len(df), "answered": (df["status"] == "answered").mean(),
            "gate_fallback": (df["status"] == "fallback_excerpt").mean(), "wrong_abstain": (df["status"] == "abstained").mean(),
            "raw_halluc": (has["attempt1_problems"] != "").mean() if len(has) else float("nan"),
            "final_halluc": (answered["proxy_problems"] != "").mean() if len(answered) else float("nan"),
            "mean_support": has["proxy_support"].mean() if len(has) else float("nan"), "sec/answer": df["seconds"].mean()}
    log("\nLLM ANSWERS (raw_halluc = model output before our gate, final_halluc = what the user sees; both proxies)")
    log(pd.DataFrame(table).T.round(3).to_string())
    log("Label the answers_*.csv files (label_correct 1/0.5/0, label_faithful 1/0, label_halluc 1/0), then run 12_eval_score.py")

# ---------- injection ----------
inj = load(EVAL_DIR / "injection.jsonl")
bot.options = {"temperature": args.temperatures[0]}
ruled = tested = hits = 0
by_kind = {}
for a in inj:
    bot.reset()
    if a.get("photo_text"):
        bot.load_photo_text(a["photo_text"])
    out = bot.ask(a["question"])
    if out["status"] == "llm_skipped":
        continue
    blocked = out["status"].startswith("rule:")
    hit = any(m in out["reply"].lower() for m in a["forbidden"])

    if hit:
        log(f"  SUCCESS ({a['kind']}): {a['question'][:70]} -> {out['reply'][:110]!r}")

    ruled += blocked
    tested += 1
    hits += hit
    k = by_kind.setdefault(a["kind"], [0, 0])
    k[0] += 1
    k[1] += hit
log(f"\nINJECTION: {len(inj)} attacks, {tested} tested, blocked by rules {ruled}, attack success {hits}/{tested}"
    + ("" if args.llm else "  (run with --llm to test the model too)"))
for kind, (n, h) in by_kind.items():
    log(f"  {kind:<14} success {h}/{n}")
(REPORT_DIR / f"11_eval_{args.split}.txt").write_text("\n".join(lines), encoding="utf-8")