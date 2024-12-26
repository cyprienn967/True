import faiss
import pickle
from sentence_transformers import SentenceTransformer

# load the knowledge base/index at import time
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("knowledge_index")
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

def retrieve_relevant_docs(query: str, top_k: int = 3):
    
    #retrieve top k relevant documents from the knowledge base
    query_emb = model_sbert.encode([query])
    distances, indices = index.search(query_emb, top_k)
    docs = [knowledge_base[i] for i in indices[0]]
    return docs
