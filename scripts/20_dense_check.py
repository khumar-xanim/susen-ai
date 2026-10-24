"""See what the embedding search finds inside ONE leaflet. Usage: python scripts\\20_dense_check.py adaklin "gunes isiginda istifade etmek olar?" """
import sys

from dense import LEAFLET_DENSE
from kb import embed
from product_matcher import ProductIndex
from retrieval import load_chunks

index, chunks = ProductIndex(), load_chunks()
med = index.find(sys.argv[1])["candidates"][0]["id"]
d = LEAFLET_DENSE.scores(med, embed([sys.argv[2]], cpu=True)[0])
for c in sorted(chunks[med], key=lambda c: -d.get(c["chunk_index"], 0))[:6]:
    print(f"{d.get(c['chunk_index'], 0):.2f}  [{c['category']}] p.{c['page_start']} {c['section'][:40]}: {c['text'][:90]}")