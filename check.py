import os, pickle
FAISS_DIR = "faiss"
store = "pdf_store"
p = os.path.join(FAISS_DIR, store, "id2text.pkl")

if not os.path.exists(p):
    print("No id2text.pkl found at", p)
    # Maybe index.pkl exists instead:
    p2 = os.path.join(FAISS_DIR, store, "index.pkl")
    print("Check index.pkl at", p2, "exists?", os.path.exists(p2))
else:
    with open(p, "rb") as f:
        id2text = pickle.load(f)
    print("Type:", type(id2text))
    # show a few keys and values (shortened)
    if isinstance(id2text, dict):
        keys = list(id2text.keys())[:10]
        print("Sample keys:", keys)
        for k in keys:
            val = id2text[k]
            print(f"ID {k} -> type {type(val)}")
            s = str(val)[:400].replace("\n", " ")
            print("  ->", s)
    else:
        # print representation summary
        print("repr:", repr(id2text)[:1000])
