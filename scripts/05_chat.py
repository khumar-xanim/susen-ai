"""Chat in the terminal. Usage: python scripts\\05_chat.py [--model qwen3.5:4b] [--source]"""
import argparse

from chatbot import Chatbot
from llm import warm_up

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--source", action="store_true", help="also print the original leaflet sentences")
args = ap.parse_args()
bot = Chatbot(model=args.model)
print(f"Loading {bot.model} ...")
try:
    print(f"model ready in {warm_up(bot.model)} s")
except Exception as e:
    print("Ollama problem:", e)
print("SUSEN AI is not a doctor or pharmacist; it reads official leaflets. Empty line or Ctrl+C quits.")
try:
    while q := input("\nSiz: ").strip():
        r = bot.ask(q)
        print(f"\nSUSEN: {r['reply']}\n   [{r['status']} | {r['seconds']} s | model {r['model_seconds']} s | attempts {r.get('attempts', '-')}]")        
        if args.source and r.get("excerpt"):
            print("   original:", r["excerpt"]["text"], "|", r["excerpt"]["label"])
except KeyboardInterrupt:
    print("\nBye")