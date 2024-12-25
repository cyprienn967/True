from sentence_transformers import SentenceTransformer
import faiss
import pickle

# load model, index, and knowledge base
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("knowledge_index")
with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

def retrieve_relevant_docs(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_base[i] for i in indices[0]]

