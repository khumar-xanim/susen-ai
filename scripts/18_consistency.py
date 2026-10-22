"""Does the same question get an answer every time? Usage: python scripts\\18_consistency.py [--n 20] [--repeat 3]"""
import argparse
import json
import random

from chatbot import Chatbot
from config import EVAL_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--repeat", type=int, default=3)
args = ap.parse_args()
qs = [json.loads(x) for x in open(EVAL_DIR / "questions_dev.jsonl", encoding="utf-8")]
qs = [q for q in qs if q["type"] == "answerable"]
random.Random(3).shuffle(qs)
bot = Chatbot()
rates, attempts, misses = [], [], []
for q in qs[:args.n]:
    ok = []
    for _ in range(args.repeat):
        bot.reset()
        out = bot.ask(q["question"])
        ok.append(out["status"] == "answered")
        attempts.append(out.get("attempts", 0))
    rates.append(sum(ok) / args.repeat)
    if sum(ok) < args.repeat:
        misses.append((q["question"], sum(ok)))
print(f"answered every time: {sum(r == 1 for r in rates)}/{len(rates)} questions; mean answer rate {sum(rates) / len(rates):.0%}")
print(f"mean generation attempts per answer: {sum(attempts) / max(len(attempts), 1):.2f}")
for question, n in misses:
    print(f"  not always answered ({n}/{args.repeat}): {question}")