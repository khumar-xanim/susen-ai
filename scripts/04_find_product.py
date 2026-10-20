"""Usage: python scripts\\04_find_product.py "exinaseya komp dermani almisam"   (no argument = interactive)"""
import sys

from product_matcher import ProductIndex

index = ProductIndex()


def show(query):
    res = index.find(query)
    print(f"\n> {query}\n  status: {res['status']} ({res['n_total']} rows)")
    for c in res["candidates"]:
        flag = "leaflet" if c["has_leaflet"] else "NO leaflet"
        print(f"   {c['id']:<38} {c['form']:<28} {c['dose']:<16} {c['price_azn']:>9} AZN  [{flag}]")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    print("Type a message (empty line quits)")
    while q := input("> ").strip():
        show(q)