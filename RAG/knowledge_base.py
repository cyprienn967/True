import os
import pickle
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text from the page
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, max_chars=500) -> list[str]:
    """
    Split text into chunks of up to max_chars, trying to chunk by paragraph.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

def build_knowledge_base_from_pdfs(pdf_folder="pdf_docs", max_chars=500):
    """
    1) Gather all PDFs in pdf_folder
    2) Extract and chunk text
    3) Build BM25 + FAISS indexes
    4) Save them to disk
    """
    documents = []

    # 1) Process each PDF
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, fname)
            # 2) Extract text
            text = extract_text_from_pdf(pdf_path)
            # 3) Chunk text
            chunked_list = chunk_text(text, max_chars=max_chars)
            # 4) Append to documents
            documents.extend(chunked_list)

    # Build embeddings for FAISS
    model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_sbert.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "knowledge_index")

    # Build BM25 index
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Save the chunked documents
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(documents, f)

    print(f"Knowledge base built with {len(documents)} chunks from PDFs in '{pdf_folder}'.")

if __name__ == "__main__":
    build_knowledge_base_from_pdfs(pdf_folder="pdf_docs", max_chars=1000)
