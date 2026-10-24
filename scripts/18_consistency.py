"""Is the same question answered every time? Usage: python scripts\\18_consistency.py [--n 20] [--repeat 3]"""
import argparse
import json
import random
from collections import Counter

from chatbot import Chatbot
from config import EVAL_DIR

RESPONDED = {"answered", "unknown_in_leaflet", "abstained"}
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--repeat", type=int, default=3)
args = ap.parse_args()
qs = [json.loads(x) for x in open(EVAL_DIR / "questions_dev.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["type"] == "answerable"]
random.Random(3).shuffle(qs)
bot = Chatbot()
always, attempts, reasons, misses = 0, [], Counter(), []
for q in qs[:args.n]:
    statuses = []
    for _ in range(args.repeat):
        bot.reset()
        out = bot.ask(q["question"])
        statuses.append(out["status"])
        if out.get("attempts"):
            attempts.append(out["attempts"])
            reasons.update(r[:55] for r in (out.get("attempt1_problems") or ["accepted at once"]))
    if all(s in RESPONDED for s in statuses):
        always += 1
    else:
        misses.append((q["question"], statuses))
print(f"responded every time: {always}/{min(args.n, len(qs))} questions")
print(f"mean generation attempts per answer: {sum(attempts) / max(len(attempts), 1):.2f}")
print("why the first attempt was rejected:", dict(reasons.most_common(6)))
for question, statuses in misses:
    print(f"  {question}  ->  {statuses}")