import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import openai

from conversation_store import conversation_store
from retrieval import hybrid_retrieve
from verification import verify_partial_answer
from critique import correct_after_critique
from pipeline import chain_of_thought_reader


openai.api_key = "" #put a key here

app = FastAPI(
    title="Stepwise RAG Verification API",
    version="1.0.0",
    description="Retrieval + Verification + Correction"
)

class VerifyRequest(BaseModel):
    conversation_id: str
    partial_text: str
    auto_correct: bool = True

class VerifyResponse(BaseModel):
    verified_text: str
    was_contradicted: bool
    conversation_id: str

class ConversationResponse(BaseModel):
    conversation_id: str
    partial_steps: list[str]


@app.post("/verify_step", response_model=VerifyResponse)
def verify_step(req: VerifyRequest):
    """
    Endpoint to verify (and optionally correct) a partial step using:
    - NLI aggregator (verify_partial_answer)
    - correction (correct_after_critique)
    """
    conversation_id = req.conversation_id
    partial_text = req.partial_text

    conversation_store.init_conversation(conversation_id)

    # 1. Verify partial step
    is_valid = verify_partial_answer(partial_text)
    was_contradicted = not is_valid
    final_text = partial_text

    # 2. If invalid => try auto-correction
    if not is_valid and req.auto_correct:
        # Retrieve relevant docs to help correction
        relevant_docs = hybrid_retrieve(partial_text, top_k=3)
        corrected_text = correct_after_critique(partial_text, relevant_docs)
        # Re-verify the corrected text
        if verify_partial_answer(corrected_text):
            final_text = corrected_text
            was_contradicted = True
        else:
            final_text = "[HALTED] Could not correct contradiction."

    # 3. Add final step to conversation
    conversation_store.add_step(conversation_id, final_text)

    return VerifyResponse(
        verified_text=final_text,
        was_contradicted=was_contradicted,
        conversation_id=conversation_id
    )

@app.get("/chain_of_thought_reader")
def chain_of_thought_reader_endpoint(query: str, top_k: int = 3):
    """
    Example endpoint demonstrating a chain-of-thought 'retriever-reader' pipeline.
    """
    answer = chain_of_thought_reader(query, top_k=top_k)
    return {"query": query, "answer": answer}


@app.get("/conversation/{conv_id}", response_model=ConversationResponse)
def get_conversation(conv_id: str):
    """
    Retrieve all partial steps for the specified conversation ID.
    """
    steps = conversation_store.get_steps(conv_id)
    return ConversationResponse(
        conversation_id=conv_id,
        partial_steps=steps
    )


if __name__ == "__main__":
    import uvicorn
    # Run uvicorn if you want to serve locally
    # uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    pass
