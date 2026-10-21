"""Score your manual labels in outputs\\eval\\answers_*.csv.
Judge the model's own words (llm_text) against evidence_text: label_correct 1 / 0.5 / 0, label_faithful 1 / 0,
label_halluc 1 (invented something) / 0."""
import glob
import sys

import pandas as pd

from config import EVAL_DIR

for f in (sys.argv[1:] or sorted(glob.glob(str(EVAL_DIR / "answers_*.csv")))):
    df = pd.read_csv(f)
    lab = df.dropna(subset=["label_correct"])
    if lab.empty:
        print(f"{f}: no labels yet")
        continue
    print(f"\n{f}: {len(lab)} labelled answers")
    print(f"  answer correctness: {lab['label_correct'].mean():.2f}")
    print(f"  faithfulness (manual): {lab['label_faithful'].mean():.2f}")
    print(f"  hallucination rate (manual): {lab['label_halluc'].mean():.1%}")
    flagged = lab["proxy_problems"].fillna("") != ""
    h = lab["label_halluc"] == 1
    tp, fp, fn = (flagged & h).sum(), (flagged & ~h).sum(), (~flagged & h).sum()
    print(f"  automatic check vs your labels: precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f}")