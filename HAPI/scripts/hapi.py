import torch
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional


from mlp_detection import HallucinationDetector

from bio_gpt_utils import BioGPTGenerator

router = APIRouter()


device = "cuda" if torch.cuda.is_available() else "cpu"
hallu_detector = HallucinationDetector(
    mlp_path="data/best_mlp_model.pt",
    device=device,
    model_name="microsoft/biogpt",
    hidden_dim=1024  
)

bio_gpt_generator = BioGPTGenerator()  

class HAPIRequest(BaseModel):
    question: str
    conversation_id: str
    with_hallucination_filter: bool = True

class HAPIResponse(BaseModel):
    raw_answer: str
    final_answer: str
    was_hallucinated: bool

@router.post("/hapi_generate", response_model=HAPIResponse)
def hapi_generate(req: HAPIRequest):
    
    question = req.question
    
    
    raw_ans = bio_gpt_generator.generate_raw_answer(question, max_length=200)
    
    final_ans = raw_ans
    was_hallucinated = False

    if req.with_hallucination_filter:
        
        is_hallu = hallu_detector.is_hallucinated(raw_ans)
        if is_hallu:
            was_hallucinated = True
            
            corrected_ans = bio_gpt_generator.generate_raw_answer(
                f"[IMPROVE ANSWER]: {question}", max_length=100
            )
            final_ans = corrected_ans
    
    return HAPIResponse(
        raw_answer=raw_ans,
        final_answer=final_ans,
        was_hallucinated=was_hallucinated
    )
