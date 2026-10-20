"""Chat in the terminal. Usage: python scripts\\05_chat.py [--model gemma3:4b]"""
import argparse

from chatbot import Chatbot

ap = argparse.ArgumentParser()
ap.add_argument("--model")
args = ap.parse_args()
bot = Chatbot(model=args.model)
print(f"SUSEN AI ready (model: {bot.model}). Empty line quits.")
while q := input("\nSiz: ").strip():
    r = bot.ask(q)
    print(f"\nSUSEN: {r['reply']}\n   [{r['status']} | {r['seconds']} s | {r['model']}]")