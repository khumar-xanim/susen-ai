"""Chat in the terminal. Usage: python scripts\\05_chat.py [--model qwen3.5:9b]"""
import argparse

from chatbot import Chatbot
from llm import warm_up

ap = argparse.ArgumentParser()
ap.add_argument("--model")
args = ap.parse_args()
bot = Chatbot(model=args.model)
print(f"Loading {bot.model} ...")
try:
    print(f"model ready in {warm_up(bot.model)} s")
except Exception as e:
    print("Ollama problem:", e)
print("SUSEN AI ready. Empty line or Ctrl+C quits.")
try:
    while q := input("\nSiz: ").strip():
        r = bot.ask(q)
        print(f"\nSUSEN: {r['reply']}\n   [{r['status']} | {r['seconds']} s | {r['model']}]")
except KeyboardInterrupt:
    print("\nBye")