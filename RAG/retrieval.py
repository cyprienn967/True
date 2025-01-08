import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re

# Load resources
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("knowledge_index")

with open("knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def preprocess_chunk(chunk: str, max_length: int = 1024) -> str:
    """
    Preprocess a chunk by cleaning and truncating it to a manageable length.
    
    Args:
        chunk (str): The raw chunk of text.
        max_length (int): Maximum character length for the chunk.

    Returns:
        str: Cleaned and truncated chunk.
    """
    # Clean up whitespace and special characters
    chunk = re.sub(r"\s+", " ", chunk.strip())

    # Truncate the chunk to a manageable length
    if len(chunk) > max_length:
        chunk = chunk[:max_length]

    return chunk

def summarize_chunks(chunks: list[str], max_tokens: int = 100) -> list[str]:
    """
    Summarizes each retrieved chunk to reduce token length while preserving key information.

    Args:
        chunks (list[str]): List of retrieved chunks.
        max_tokens (int): Maximum tokens for each summarized chunk.

    Returns:
        list[str]: List of summarized chunks.
    """
    summarized_chunks = []
    for chunk in chunks:
        try:
            # Preprocess the chunk before summarization
            chunk = preprocess_chunk(chunk)

            # Summarize the chunk using the summarizer
            summary = summarizer(chunk, max_length=max_tokens, min_length=50, truncation=True)
            summarized_chunks.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"[WARNING] Summarization failed for chunk: {chunk[:100]}... Error: {e}")
            
            # Fallback: Use preprocessed chunk instead
            summarized_chunks.append(chunk)
    
    return summarized_chunks


def clean_chunks(chunks: list[str]) -> list[str]:

    clean_chunks_list = []
    for chunk in chunks:
        clean_chunk = re.sub(r"\s+", " ", chunk.strip())
        # Example of removing specific noise patterns if needed:
        clean_chunk = re.sub(r"(Ill-health and death from swine flu:)+", "", clean_chunk)
        # Remove any leftover HTML-like tokens
        clean_chunk = re.sub(r"<[^>]+>", "", clean_chunk)
        clean_chunks_list.append(clean_chunk)
    return clean_chunks_list


def hybrid_retrieve(query: str, top_k: int = 5, max_summary_tokens: int = 150):
    """
    Hybrid retrieval combining BM25 and FAISS embeddings.
    Includes summarization of the retrieved chunks to reduce token count.
    
    Args:
        query (str): Query string.
        top_k (int): Number of top documents to retrieve.
        max_summary_tokens (int): Maximum tokens for each summarized chunk.

    Returns:
        list[str]: List of summarized chunks.
    """
    # Perform BM25 retrieval
    query_terms = query.split(" ")
    bm25_scores = bm25.get_scores(query_terms)

    # Perform FAISS embedding-based retrieval
    query_emb = model_sbert.encode([query])
    distances, indices = index.search(query_emb, len(knowledge_base))
    distances = distances[0]
    indices = indices[0]

    # Convert distances to similarity
    similarity = 1.0 / (1.0 + distances)

    # Combine BM25 and FAISS scores
    combined_scores = {}
    for doc_idx, bm25_s in enumerate(bm25_scores):
        faiss_pos = np.where(indices == doc_idx)[0]
        embed_s = similarity[faiss_pos[0]] if len(faiss_pos) > 0 else 0
        combined_scores[doc_idx] = bm25_s + (embed_s * 2.0)

    # Sort by combined score
    sorted_doc_idxs = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)

    # Retrieve top_k chunks
    top_docs = [knowledge_base[i] for i in sorted_doc_idxs[:top_k]]

    # Clean and summarize the retrieved chunks
    clean_docs = clean_chunks(top_docs)
    summarized_docs = summarize_chunks(clean_docs, max_tokens=max_summary_tokens)

    return summarized_docs
