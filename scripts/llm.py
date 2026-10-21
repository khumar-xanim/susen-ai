"""The only file that talks to a language model. Change model with SUSEN_MODEL."""
import re
import time

import ollama

from config import LLM_PROVIDER, LLM_MODEL, LLM_OPTIONS

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
KEEP_ALIVE = "30m"      # keep the model in memory between questions


def _chat(model, messages, options):
    if LLM_PROVIDER != "ollama":
        raise NotImplementedError(f"Provider '{LLM_PROVIDER}' is not implemented yet")
    try:
        return ollama.chat(model=model, messages=messages, options=options, think=False, keep_alive=KEEP_ALIVE)
    except Exception as e:
        if "think" not in str(e).lower():      # some models do not accept the think flag
            raise
        return ollama.chat(model=model, messages=messages, options=options, keep_alive=KEEP_ALIVE)


def warm_up(model=None):
    """Load the model into memory so the first real question is not slow."""
    t0 = time.time()
    _chat(model or LLM_MODEL, [{"role": "user", "content": "Salam"}], {**LLM_OPTIONS, "num_predict": 1})
    return round(time.time() - t0, 1)


def generate(messages, model=None):
    t0 = time.time()
    resp = _chat(model or LLM_MODEL, messages, LLM_OPTIONS)
    text = THINK_BLOCK.sub("", resp["message"]["content"]).strip()
    return text, round(time.time() - t0, 1)