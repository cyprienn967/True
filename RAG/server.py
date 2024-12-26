# server.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import openai

from conversation_store import conversation_store
from verification import verify_partial_answer
from retrieval import retrieve_relevant_docs


#  Set your OpenAI API Key

openai.api_key = "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"


#   FastAPI Initialization

app = FastAPI(
    title="Stepwise RAG Verification API",
    version="1.0.0",
    description="Provides endpoints for verifying partial steps in a conversation using RAG & NLI stuff"
)


class VerifyRequest(BaseModel):
    conversation_id: str
    partial_text: str
    auto_correct: bool = True  # if True, attempt auto-correction if verification fails

class VerifyResponse(BaseModel):
    verified_text: str
    was_contradicted: bool
    conversation_id: str

class ConversationResponse(BaseModel):
    conversation_id: str
    partial_steps: list[str]


#   Auto-correction function

def correct_step_with_openai(original_text: str, references: list[str]) -> str:

    system_prompt = (
        "You are a factual correctness assistant. You generated a partial answer that contradicts known facts.\n"
        "Below are the relevant references from your knowledge base:\n" +
        "\n".join(f"- {r}" for r in references) + "\n" +
        "Please rewrite the partial answer so that it is factually correct and aligns with the references.\n"
        "Your answer should be concise (1-2 sentences)."
    )
    user_prompt = f"Original partial answer: {original_text}\nRewrite it correctly."

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=100,
            temperature=0.5
        )
        corrected = response.choices[0].message.content.strip()
        print("DEBUG: Correction function was called")
        print("DEBUG: Original partial text:", original_text)
        print("DEBUG: References passed in:", references)
        print("DEBUG: The LLM responded with corrected text:", corrected)
        return corrected
    except Exception as e:
        return f"[Correction Error: {str(e)}]"


#   POST /verify_step

@app.post("/verify_step", response_model=VerifyResponse)
def verify_step(req: VerifyRequest):
    
    #endpoint that LLM stores after generating a partial step, if contradiction is found and auto_correct is True we correct
    #store resulting text in convo store
    conversation_id = req.conversation_id
    partial_text = req.partial_text


    conversation_store.init_conversation(conversation_id)

    is_valid = verify_partial_answer(partial_text)
    was_contradicted = not is_valid
    final_text = partial_text

    if not is_valid:
        if req.auto_correct:
            
            relevant_refs = retrieve_relevant_docs(partial_text, top_k=3)
            corrected_text = correct_step_with_openai(partial_text, relevant_refs)
            
            if verify_partial_answer(corrected_text):
                final_text = corrected_text
                was_contradicted = True
            else:
                
                final_text = "[HALTED] Could not correct contradiction."
        else:
            final_text = "[FAILED VERIFICATION]"

    
    conversation_store.add_step(conversation_id, final_text)

    return VerifyResponse(
        verified_text=final_text,
        was_contradicted=was_contradicted,
        conversation_id=conversation_id
    )


#  GET /conversation/{conv_id}

@app.get("/conversation/{conv_id}", response_model=ConversationResponse)
def get_conversation(conv_id: str):
    """
    Retrieve all partial steps for a given conversation ID.
    """
    steps = conversation_store.get_steps(conv_id)
    return ConversationResponse(
        conversation_id=conv_id,
        partial_steps=steps
    )


#  For local running

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

