from sentence_transformers import SentenceTransformer
import faiss
import pickle

#knowledge base its pretty small type shit but like
knowledge_base = [
    "The capital of France is Paris.",
    "The moon is Earth's only natural satellite.",
    "Mars is uninhabited and has no population."
]

#creat embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(knowledge_base)

# save embeddings and the model for retrieval
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

#save index n knowledge base
faiss.write_index(index, "knowledge_index")
with open("knowledge_base.pkl", "wb") as f:
    pickle.dump(knowledge_base, f)
