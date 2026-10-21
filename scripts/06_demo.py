"""Run questions from a text file (blank line = new conversation, '#' lines are labels).
Usage: python scripts\\06_demo.py --file scripts\\manual_questions.txt [--models qwen3.5:4b gemma3:4b] [--temperature 0.1]"""
import argparse

from chatbot import Chatbot
from config import OUT_DIR, LLM_MODEL
from llm import warm_up

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="scripts/manual_questions.txt")
ap.add_argument("--models", nargs="+", default=[LLM_MODEL])
ap.add_argument("--temperature", type=float, default=0.1)
args = ap.parse_args()

convs, cur, label = [], [], ""
for raw in open(args.file, encoding="utf-8"):
    line = raw.strip()
    if line.startswith("#"):
        label = line.lstrip("# ").strip()
    elif line:
        cur.append((label, line))
    elif cur:
        convs.append(cur)
        cur = []
if cur:
    convs.append(cur)

bot = Chatbot(temperature=args.temperature)
md = []
for model in args.models:
    bot.model = model
    print(f"loading {model}: {warm_up(model)} s")
    md.append(f"# Model: {model} (temperature {args.temperature})\n")
    n = 0
    for conv in convs:
        bot.reset()
        for label, q in conv:
            n += 1
            r = bot.ask(q)
            block = (f"### {n}. {label}\n\n**Siz:** {q}\n\n**SUSEN:** {r['reply']}\n\n*{r['status']} | {r['seconds']} s*\n\n"
                     "**Sizin qiymətləndirmə:** ☐ düzgün  ☐ qismən  ☐ səhv  ☐ uydurma    Qeyd:\n\n---\n")
            print(block)
            md.append(block)
(OUT_DIR / "demo_answers.md").write_text("\n".join(md), encoding="utf-8")
print("Saved:", OUT_DIR / "demo_answers.md")