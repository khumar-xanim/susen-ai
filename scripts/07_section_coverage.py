"""Which leaflet headings are not mapped to a section category yet?"""
import json
from collections import Counter

from config import CHUNKS_FILE
from retrieval import category_of

cats, other = Counter(), Counter()
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        k = category_of(c["section"])
        cats[k] += 1
        if k == "other":
            other[c["section"]] += 1
total = sum(cats.values())
print(f"chunks: {total}")
for k, n in cats.most_common():
    print(f"  {k:<20}{n:>8}  {n / total:.1%}")
print("\nTop 60 unmapped headings:")
for h, n in other.most_common(60):
    print(f"{n:>6}  {h[:110]}")