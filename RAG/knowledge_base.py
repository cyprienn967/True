import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def build_knowledge_base():
    """
    Builds a larger knowledge base with both BM25 and FAISS indexes.
    """

    documents = [
        "The capital of France is Paris.",
        "The moon is Earth's only natural satellite.",
        "Mars is uninhabited and has no population.",
        "Python is a popular programming language created by Guido van Rossum.",
        "OpenAI created GPT models for language understanding.",
        "Faiss is a library that supports efficient vector similarity search.",
        "BM25 is a classical retrieval algorithm based on term frequency.",
        "JavaScript was originally developed by Brendan Eich.",
        "Java was originally developed by James Gosling at Sun Microsystems.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of the United Kingdom is London.",
        "The largest planet in our Solar System is Jupiter.",
        "Saturn has the most extensive ring system of any planet.",
        "Neptune is the farthest planet from the Sun in our Solar System.",
        "Albert Einstein developed the theory of relativity.",
        "The speed of light in vacuum is approximately 299,792 km/s.",
        "The official language of Brazil is Portuguese.",
        "Canada is the second-largest country by land area.",
        "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas.",
        "Bananas are a fruit that are rich in potassium.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The capital of Japan is Tokyo.",
        "K2 is the second-highest mountain on Earth, after Mount Everest.",
        "The Taj Mahal is located in Agra, India.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "Vincent van Gogh painted The Starry Night.",
        "The Titanic sank in 1912 during its maiden voyage.",
        "Shakespeare wrote 'Romeo and Juliet'.",
    ]

    # Save the final knowledge base for reference
    knowledge_base = documents

    # 1. Build embeddings for FAISS
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(knowledge_base)

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "knowledge_index")

    # 2. Save the knowledge base text in a pickle
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)

    # 3. Create a BM25 index
    tokenized_corpus = [doc.split(" ") for doc in knowledge_base]
    bm25 = BM25Okapi(tokenized_corpus)

    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("Knowledge base, FAISS index, and BM25 index have been created.")

if __name__ == "__main__":
    if not os.path.exists("knowledge_index") or not os.path.exists("knowledge_base.pkl"):
        build_knowledge_base()
    else:
        print("Knowledge base already exists. Delete files if you want to rebuild.")
