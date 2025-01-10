# File: HAPI/scripts/hapi.py

import logging
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from mlp_detection import HallucinationDetector
from bio_gpt_utils import BioGPTGenerator

router = APIRouter()
device = "cuda" if torch.cuda.is_available() else "cpu"

hallu_detector = HallucinationDetector(
    mlp_path="../data/best_mlp_model.pt",
    device=device,
    model_name="microsoft/biogpt",
    hidden_dim=1024
)
bio_gpt_generator = BioGPTGenerator()

class HAPIRequest(BaseModel):
    question: str
    with_hallucination_filter: bool = True

class HAPIResponse(BaseModel):
    raw_answer: str
    final_answer: str
    was_hallucinated: bool

@router.post("/hapi_generate", response_model=HAPIResponse)
def hapi_generate(req: HAPIRequest):
    """
    Generate from BioGPT -> MLP check. If hallucinated => second pass with a different prompt.
    We do not slice out the prompt from the final decode, so you get the *full* text each time.
    """
    question = req.question

    # We'll add a simple "Answer:" to help instruct the model to produce an actual answer.
    prompt_raw = f"{question}\nYou are an AI trained to provide concise, factual answers to biomedical questions. Avoid enumerations, references, and placeholder text. Focus solely on delivering clear and direct responses.:"

    try:
        raw_ans = bio_gpt_generator.generate_raw_answer(prompt_raw, max_new_tokens=400)
    except Exception as e:
        print("[ERROR] Generation (raw) failed:", str(e))
        raw_ans = ""

    final_ans = raw_ans
    was_hallucinated = False

    if req.with_hallucination_filter:
        try:
            is_hallu = hallu_detector.is_hallucinated(raw_ans)
            if is_hallu:
                was_hallucinated = True
                # second pass with a slightly more directive prompt
                # ensure max_new_tokens is large as well
                prompt_hapi = f"Reanswer more succinctly:\n{question}\nYou are an AI trained to provide concise, factual answers to biomedical questions. Avoid enumerations, references, and placeholder text. Focus solely on delivering clear and direct responses."
                improved = bio_gpt_generator.generate_raw_answer(prompt_hapi, max_new_tokens=400)
                final_ans = improved
        except Exception as e:
            print("[ERROR] Hallucination check or second generation failed:", str(e))
            was_hallucinated = False

    return HAPIResponse(
        raw_answer=raw_ans,
        final_answer=final_ans,
        was_hallucinated=was_hallucinated
    )
