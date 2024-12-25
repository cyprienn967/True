from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import retrieve_relevant_docs

app = FastAPI()

class TextPayload(BaseModel):
    generated_text: str

@app.post("/check_hallucination")
def check_hallucination(payload: TextPayload):
    generated_text = payload.generated_text
    relevant_docs = retrieve_relevant_docs(generated_text, top_k=3)
    
    # Simple exact match or semantic comparison
    if any(doc in generated_text for doc in relevant_docs):
        return {"hallucinations": False, "relevant_docs": relevant_docs}
    else:
        return {"hallucinations": True, "relevant_docs": relevant_docs}



