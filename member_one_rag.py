import os
import numpy as np
import faiss
import pickle
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
# Local embedding models
# =======================
print("🔧 Loading local embedding models...")
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # 768-dim
minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim
print("✅ Local models loaded successfully")

# =======================
# Utility function for embedding
# =======================
def get_embedding(text: str, target_dim: int):
    try:
        model_emb = mpnet_model if target_dim == 768 else minilm_model
        emb = model_emb.encode(text, normalize_embeddings=True)
        emb = np.array(emb, dtype=np.float32)
        return emb
    except Exception as e:
        print(f"⚠️ Local embedding error: {e}")
        return np.zeros((target_dim,), dtype=np.float32)

# =======================
# FAISS Store Loading
# =======================
FAISS_DIR = "faiss"
store_configs = {
    "pdf_store": 768,
    "csv_store": 768,
    "json_store": 768,
    # "url_store": 384,  # Removed
}

stores = {}

for store_name, dim in store_configs.items():
    store_path = os.path.join(FAISS_DIR, store_name)
    faiss_index_path = os.path.join(store_path, "index.faiss")
    pkl_path = os.path.join(store_path, "index.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(pkl_path):
        try:
            index = faiss.read_index(faiss_index_path)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            stores[store_name] = {"index": index, "data": data, "dim": dim}
            print(f"✅ Loaded FAISS index for {store_name}")
        except Exception as e:
            print(f"⚠️ Error loading {store_name}: {e}")
    else:
        print(f"⚠️ Missing index files for {store_name}")

# =======================
# FastAPI Setup
# =======================
app = FastAPI(title="CyberTrace RAG Chatbot")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# =======================
# RAG Retrieval Function
# =======================
def retrieve_from_store(query_text, top_k=3):
    retrieved_docs = []

    for store_name, store_info in stores.items():
        try:
            index = store_info["index"]
            dim = store_info["dim"]
            data = store_info["data"]

            # Handle tuple or dict data
            if isinstance(data, tuple):
                docs_obj = data[0]
                if hasattr(docs_obj, "_dict"):
                    docs = list(docs_obj._dict.values())
                else:
                    docs = list(docs_obj)
            elif isinstance(data, dict):
                docs = list(data.values())
            else:
                docs = list(data)

            query_vector = get_embedding(query_text, dim)
            D, I = index.search(np.array([query_vector]), top_k)

            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(docs):
                    doc = docs[idx]
                    text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else str(doc)
                    source = getattr(doc, "metadata", {}).get("source_file", "unknown") \
                             if hasattr(doc, "metadata") else doc.get("source_file", "unknown") \
                             if isinstance(doc, dict) else "unknown"

                    retrieved_docs.append({
                        "source": store_name,
                        "source_file": source,
                        "content": text,
                        "score": float(score)
                    })
        except Exception as e:
            print(f"⚠️ Retrieval error in {store_name}: {e}")

    retrieved_docs = sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)
    return retrieved_docs[:top_k]

# =======================
# LLM Response Generator
# =======================
def generate_answer(context, query):
    try:
        prompt = f"""
        You are a cybersecurity analyst. Based on the retrieved context below, answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer clearly and concisely:display any ID or Tile mentioned in any contexts
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ LLM generation error: {e}")
        return "(LLM generation failed)"

# =======================
# Query Endpoint
# =======================
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    print(f"🔍 Query received: {request.query}")

    retrieved_docs = retrieve_from_store(request.query, top_k=request.top_k)
    context = "\n\n".join([doc["content"] for doc in retrieved_docs if doc["content"]])

    answer = generate_answer(context, request.query) if context else "(No relevant context found)"
    
    return JSONResponse({
        "query": request.query,
        "answer": answer,
        "retrieved_docs": retrieved_docs
    })

# =======================
# Root Route
# =======================
@app.get("/")
def root():
    return {"message": "🚀 CyberTrace RAG Chatbot is running successfully!"}
