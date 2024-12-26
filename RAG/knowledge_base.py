
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def build_knowledge_base():
    # builds a small knowledge base and a FAISS index for it
    knowledge_base = [
        "The capital of France is Paris.",
        "The moon is Earth's only natural satellite.",
        "Mars is uninhabited and has no population.",
        "Python is a popular programming language."
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(knowledge_base)

    # create a flat L2 index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # write index & knowledge base to disk
    faiss.write_index(index, "knowledge_index")
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

if __name__ == "__main__":
    build_knowledge_base()
    print("Knowledge base and FAISS index have been created.")
