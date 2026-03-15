import os
import numpy as np
import faiss
import pickle
import json
from datetime import datetime

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# =======================
# Load environment variables
# =======================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# =======================
# Initialize Gemini model
# =======================
model = genai.GenerativeModel("gemini-2.5-flash")

# =======================
# Load embedding model
# =======================
print("🔧 Loading embedding model...")
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("✅ Embedding model loaded")

def get_embedding(text: str):
    emb = mpnet_model.encode(text, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)

# =======================
# Load FAISS PDF Store
# =======================
FAISS_DIR = "faiss"
stores = {}

pdf_dir = os.path.join(FAISS_DIR, "pdf_store")
pdf_index_file = os.path.join(pdf_dir, "index.faiss")
pdf_text_file = os.path.join(pdf_dir, "id2text.pkl")
pdf_meta_file = os.path.join(pdf_dir, "id2meta.pkl")

if not (os.path.exists(pdf_index_file)
        and os.path.exists(pdf_text_file)
        and os.path.exists(pdf_meta_file)):
    raise FileNotFoundError("❌ PDF FAISS store files missing")

pdf_index = faiss.read_index(pdf_index_file)

with open(pdf_text_file, "rb") as f:
    pdf_id2text = pickle.load(f)

with open(pdf_meta_file, "rb") as f:
    pdf_id2meta = pickle.load(f)

stores["pdf_store"] = {
    "index": pdf_index,
    "id2text": pdf_id2text,
    "id2meta": pdf_id2meta
}

print("✅ PDF store loaded successfully")

# =======================
# Load CSV Store
# =======================
csv_dir = os.path.join(FAISS_DIR, "csv_store")
csv_index_file = os.path.join(csv_dir, "index.faiss")
csv_pkl_file = os.path.join(csv_dir, "index.pkl")

if os.path.exists(csv_index_file) and os.path.exists(csv_pkl_file):
    csv_index = faiss.read_index(csv_index_file)
    with open(csv_pkl_file, "rb") as f:
        csv_docstore, csv_id_map = pickle.load(f)

    stores["csv_store"] = {
        "index": csv_index,
        "docstore": csv_docstore,
        "id_map": csv_id_map
    }
    print("✅ Loaded CSV store")



# =======================
# Load URL Store
# =======================
url_dir = os.path.join(FAISS_DIR, "url_store")
url_index_file = os.path.join(url_dir, "cyber_vectors.index")
url_meta_file = os.path.join(url_dir, "cyber_metadata.json")

if os.path.exists(url_index_file) and os.path.exists(url_meta_file):
    url_index = faiss.read_index(url_index_file)
    with open(url_meta_file, "r", encoding="utf-8") as f:
        url_metadata = json.load(f)

    stores["url_store"] = {
        "index": url_index,
        "metadata": url_metadata
    }
    print("✅ Loaded URL store")


# =======================
# FastAPI Setup
# =======================
app = FastAPI(title="CyberTrace – RAG Backend (PDF Only)")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# =======================
# Time Utilities
# =======================
def parse_timestamp(ts):
    try:
        return datetime.fromisoformat(ts).timestamp()
    except:
        return 0.0

def time_aware_rerank(docs, alpha=0.7):
    """
    alpha = semantic importance
    (1-alpha) = recency importance
    """
    if not docs:
        return docs

    max_time = max(parse_timestamp(d.get("timestamp", "")) for d in docs) or 1.0

    for d in docs:
        semantic_score = d["score"]
        recency_score = parse_timestamp(d.get("timestamp", "")) / max_time
        d["final_score"] = alpha * semantic_score + (1 - alpha) * recency_score

    docs.sort(key=lambda x: x["final_score"], reverse=True)
    return docs

# =======================
# Retrieval Function (PDF + Time-Aware)
# =======================
def retrieve_from_all_stores(query_text, top_k=3):
    all_docs = []

    query_vector = get_embedding(query_text)
    query_vector = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # ---------- PDF ----------

    if "pdf_store" in stores:
        s = stores["pdf_store"]
        if s["index"].d == query_vector.shape[1]:
            D, I = s["index"].search(query_vector, top_k * 3)

            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue
                meta = s["id2meta"].get(int(idx), {})
                all_docs.append({
                    "source": "pdf",
                    "file": meta.get("source_file"),
                    "page": meta.get("page"),
                    "timestamp": meta.get("timestamp", ""),
                    "content": s["id2text"].get(int(idx), ""),
                    "score": float(score)
                })

    # ---------- CSV ----------
    if "csv_store" in stores:
        s = stores["csv_store"]
        if s["index"].d == query_vector.shape[1]:
            D, I = s["index"].search(query_vector, top_k * 2)

            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue

                doc_id = s["id_map"].get(idx)
                if not doc_id:
                    continue

                doc = s["docstore"]._dict.get(doc_id)
                if not doc:
                    continue

                all_docs.append({
                    "source": "csv",
                    "file": doc.metadata.get("source"),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    "content": doc.page_content,
                    "score": float(score)
                })
        else:
            print("⚠️ Skipping CSV store due to embedding dimension mismatch")



    # ---------- URL ----------
    if "url_store" in stores:
        s = stores["url_store"]
        if s["index"].d == query_vector.shape[1]:
            D, I = s["index"].search(query_vector, top_k * 2)

            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue

                # metadata is a LIST, not dict
                if idx >= len(s["metadata"]):
                    continue

                meta = s["metadata"][idx]

                all_docs.append({
                    "source": "url",
                    "file": meta.get("url"),
                    "timestamp": meta.get("published_date", ""),
                    "content": meta.get("text", ""),
                    "score": float(score)
                })
        else:
            print("⚠️ Skipping URL store due to embedding dimension mismatch")


    
    # 🔥 Time-aware re-ranking
    reranked = time_aware_rerank(all_docs)

    return reranked[:top_k]


# =======================
# Prompt Construction (RAG)
# =======================
def build_prompt(docs, query):
    context_block = ""
    for i, d in enumerate(docs[:5]):  # Top 5 only
        source_name = d.get("source_file") or d.get("file") or "unknown"
        timestamp = d.get("timestamp", "N/A")
        content = d.get("content", "")[:800]  # 800 char limit
        
        context_block += f"""
[Context {i+1}]
Source: {source_name}
Timestamp: {timestamp}
Content: {content}
"""
    
    return f"""You are a cybersecurity analyst. Answer in clear, professional English (3-5 sentences) using ONLY these top contexts. Use partial matches.

{context_block}

Question: {query}

Answer:"""

# =======================
# LLM Response Generator
# =======================
def generate_answer(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("⚠️ LLM Error:", e)
        return "(LLM generation failed)"

# =======================
# Query Endpoint
# =======================
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    print(f"🔍 Query received: {request.query}")
    retrieved_docs = retrieve_from_all_stores(
    request.query,
    top_k=request.top_k
    )


    if not retrieved_docs:
        return JSONResponse({
            "query": request.query,
            "answer": "(No relevant context found)",
            "contexts": []
        })

    prompt = build_prompt(retrieved_docs, request.query)
    answer = generate_answer(prompt)

    return JSONResponse({
    "query": request.query,
    "answer": answer,
    "contexts": [
        {
            "source": d.get("source", "unknown"),
            "file": d.get("source_file") or d.get("file") or "unknown",
            "page": d.get("page"),
            "timestamp": d.get("timestamp", "N/A"),
            "semantic_score": d.get("score"),
            "final_score": d.get("final_score"),
            "snippet": d.get("content", "")[:300]
        }
        for d in retrieved_docs
    ]
})


# =======================
# Root Endpoint
# =======================
@app.get("/")
def root():
    return {
        "message": "🚀 CyberTrace RAG Backend (PDF + Time-Aware Retrieval) is running"
    }