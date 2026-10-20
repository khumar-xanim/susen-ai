"""The only file that talks to a language model. Change model with SUSEN_MODEL."""
import re
import time

import ollama

from config import LLM_PROVIDER, LLM_MODEL, LLM_OPTIONS

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def generate(messages, model=None):
    model = model or LLM_MODEL
    if LLM_PROVIDER != "ollama":
        raise NotImplementedError(f"Provider '{LLM_PROVIDER}' is not implemented yet")
    t0 = time.time()
    try:
        resp = ollama.chat(model=model, messages=messages, options=LLM_OPTIONS, think=False)
    except Exception as e:
        if "think" not in str(e).lower():      # some models do not accept the think flag
            raise
        resp = ollama.chat(model=model, messages=messages, options=LLM_OPTIONS)
    text = THINK_BLOCK.sub("", resp["message"]["content"]).strip()
    return text, round(time.time() - t0, 1)