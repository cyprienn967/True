import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load resources
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("knowledge_index")

with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

def hybrid_retrieve(query: str, top_k: int = 3):
    """
    Hybrid retrieval combining BM25 and FAISS embeddings.
    Returns the top_k combined results by a simple scoring fusion.
    """

    # 1. BM25 retrieval
    query_terms = query.split(" ")
    bm25_scores = bm25.get_scores(query_terms)  # array of length = number_of_docs

    # 2. Embedding-based retrieval
    query_emb = model_sbert.encode([query])
    distances, indices = index.search(query_emb, len(knowledge_base))
    # distances shape: (1, #docs); indices shape: (1, #docs)
    distances = distances[0]
    indices = indices[0]

    # Convert L2 distances to similarity
    similarity = 1.0 / (1.0 + distances)

    # 3. Combine BM25 + embedding similarity
    combined_scores = {}
    for doc_idx, bm25_s in enumerate(bm25_scores):
        # Find doc_idx in FAISS ranking
        faiss_pos = np.where(indices == doc_idx)[0]
        if len(faiss_pos) > 0:
            embed_s = similarity[faiss_pos[0]]
        else:
            embed_s = 0
        # Weighted sum
        combined_scores[doc_idx] = bm25_s + (embed_s * 2.0)

    # 4. Sort docs by combined score
    sorted_doc_idxs = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)

    # 5. Return top_k docs
    top_docs = [knowledge_base[i] for i in sorted_doc_idxs[:top_k]]
    return top_docs
