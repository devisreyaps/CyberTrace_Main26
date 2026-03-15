import os
import numpy as np
import faiss
import pickle
import json
from datetime import datetime
import re  # Add this line!

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi import UploadFile, File, HTTPException
import tempfile
import os
from pathlib import Path

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
        # ---------- CSV ----------
        # ---------- CSV ----------
        # ---------- CSV ----------
    if "csv_store" in stores:
        s = stores["csv_store"]
        if s["index"].d == query_vector.shape[1]:
            D, I = s["index"].search(query_vector, top_k * 5)

            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue

                doc_id = s["id_map"].get(idx)
                if not doc_id:
                    continue

                doc = s["docstore"]._dict.get(doc_id)
                if not doc:
                    continue

                content_full = doc.page_content
                source_file = "CSV_Report"
                
                # Priority 1: CVE-
                if "CVE-" in content_full:
                    parts = content_full.split("CVE-")[1].split()
                    if parts:
                        source_file = "CVE-" + parts[0]
                
                # Priority 2: title: (FIXED)
                elif "title:" in content_full.lower():
                    title_start = content_full.lower().find("title:") + 6
                    title_end = content_full.find("\n", title_start)  # ✅ "\\n" → "\n"
                    if title_end == -1:
                        title_end = len(content_full)
                    source_file = content_full[title_start:title_end].strip()
                
                # Priority 3: First line (FIXED)
                else:
                    lines = [line.strip() for line in content_full.split("\n") if line.strip()]  # ✅ "\\n" → "\n"
                    for line in lines:
                        if len(line) > 15 and not line.startswith(("...", "-", "•")):
                            source_file = line[:60]
                            break

                # REAL TIMESTAMP (NO DEFAULT)
                timestamp = doc.metadata.get("timestamp", "N/A")
                if "publishedDate:" in content_full:
                    date_str = content_full.split("publishedDate:")[1].split("T")[0].strip()
                    timestamp = date_str
                elif "lastModifiedDate:" in content_full:
                    date_str = content_full.split("lastModifiedDate:")[1].split("T")[0].strip()
                    timestamp = date_str

                all_docs.append({
                    "source": "csv",
                    "source_file": source_file,
                    "file": source_file,
                    "timestamp": timestamp,  # Real OR "N/A"
                    "content": doc.page_content,
                    "score": float(score)
                })


    # ---------- URL ----------
        # ---------- URL ----------
        # ---------- URL ----------
    if "url_store" in stores:
        s = stores["url_store"]
        if s["index"].d == query_vector.shape[1]:
            D, I = s["index"].search(query_vector, top_k * 2)
            
            url_contents = {}
            for idx, score in zip(I[0], D[0]):
                if idx < 0 or idx >= len(s["metadata"]):
                    continue
                meta = s["metadata"][idx]
                
                # 🚨 CHECK IF URL ACTUALLY RELEVANT (keyword match)
                content_lower = meta.get("content", "").lower()
                query_lower = query_text.lower()
                if any(word in content_lower for word in query_lower.split()):
                    url_key = meta.get("url")
                    if url_key not in url_contents:
                        url_contents[url_key] = []
                    url_contents[url_key].append({
                        "content": meta.get("content", ""),
                        "score": float(score)
                    })
            
            # 🎯 ONLY 1 URL MAX + HARSH PENALTY
            url_count = 0
            for url_key, chunks in url_contents.items():
                if url_count >= 1:  # MAX 1 URL
                    break
                combined = " ".join([c["content"] for c in chunks[:2]])[:400]
                
                # BULLETPROOF SCORING
                raw_score = min(c["score"] for c in chunks)
                best_score = min(0.85, raw_score) * 1.8  # CAP + PENALTY
                
                all_docs.append({
                    "source": "url",
                    "source_file": url_key[:80] + "...",
                    "file": url_key,
                    "timestamp": chunks[0].get("published_date", ""),
                    "content": combined,
                    "score": best_score
                })
                url_count += 1
        else:
            print("⚠️ URL dimension mismatch - skipping")



    
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
    
    return f"""You are a cybersecurity analyst. Answer in clear, professional English (3-5 sentences) using ONLY these top contexts. 

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
# ========== MALWARE ANALYSIS ENDPOINT ==========
# ========== MALWARE ENDPOINT (Base64 - NO dependencies) ==========
@app.post("/analyze_file")
async def analyze_file_endpoint(data: dict):
    try:
        import base64
        from malware.main import analyze_file
        import tempfile
        import os
        
        file_data = data.get("file_data", "")
        filename = data.get("filename", "unknown")
        
        file_bytes = base64.b64decode(file_data)
        if len(file_bytes) > 50 * 1024 * 1024:
            return {"success": False, "error": "File too large (50MB)"}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as temp_file:
            temp_path = temp_file.name
            temp_file.write(file_bytes)
        
        result = analyze_file(temp_path)
        os.unlink(temp_path)
        
        return {
            "success": True,
            "filename": filename,
            "analysis": result,
            "is_malicious": result.get("threatresult", {}).get("malicious", 0) > 0
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "services": ["RAG", "Malware Analysis"]}



# =======================
# Root Endpoint
# =======================
@app.get("/")
def root():
    return {
        "message": "🚀 CyberTrace RAG Backend (PDF + Time-Aware Retrieval) is running"
    }