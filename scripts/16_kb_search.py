"""Try the knowledge base. Usage: python scripts\\16_kb_search.py "autoimmun xestelikleri ne demekdir" "bu gun hava nece olacaq" """
import sys

from kb import KnowledgeBase

kb = KnowledgeBase()
if not kb.ok:
    raise SystemExit("Knowledge base not built: run scripts\\15_kb_build.py")
for q in sys.argv[1:] or ["autoimmun xestelikleri ne demekdir", "etirli sular nece hazirlanir", "bu gun hava nece olacaq"]:
    print("\n>", q)
    for h in kb.search(q, k=5, min_score=0.0):
        print(f"  {h['score']:.2f}  [{h['source']}] {h['title']} p.{h.get('page') or '-'}: {h['text'][:90]}")