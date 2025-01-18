# File: HAPI/scripts/bio_gpt_utils.py

import logging
import transformers
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from suppress_roberta_warning import suppress_roberta_warning

logging.getLogger("transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

def post_process_answer(ans: str) -> str:
    """
    A light post-processing to remove repeated numeric placeholders like '(1)(2)' etc.
    We'll use a simple regex to remove bracketed digits or repeated sequences.
    For instance: (1) -> ''  or (1). -> ''
    This also removes placeholders like (†), (‡).
    """
    # Remove patterns like (1), (2), (1)., (20).
    ans = re.sub(r"\(\d+\)\.?", "", ans)
    # Remove weird references like (†), (‡), (Words)
    ans = re.sub(r"\([\w\d]+?\)", "", ans)
    # Remove repeated punctuation
    ans = re.sub(r"\.\s*\.", ".", ans)
    # Trim extra spaces
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans

class BioGPTGenerator:
    def __init__(self, model_name="microsoft/biogpt"):
        with suppress_roberta_warning():
            print(f"[BioGPTGenerator] Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_raw_answer(self, prompt: str, max_new_tokens=400):
        """
        1) We apply do_sample with repetition_penalty=1.3 to reduce repeated enumerations.
        2) We'll decode the entire text, then do a tiny post-processing to remove leftover 
           bracketed digits or references.
        3) We do not slice out the prompt here. We'll let benchmark.py handle 
           removing the prompt from the final text for metrics.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        with torch.no_grad():
            with suppress_roberta_warning():
                outputs = self.model.generate(
                    **inputs,
                    min_new_tokens=50,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    temperature=0.4,
                    repetition_penalty=1.3
                )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove mention of truncated abstracts if present
        full_text = full_text.replace("(ABSTRACT TRUNCATED AT 250 WORDS)", "")
        full_text = full_text.replace("(ABSTRACT TRUNCATED AT 400 WORDS)", "")

        # Light post-processing to remove repeated numeric placeholders, etc.
        final_answer = post_process_answer(full_text)
        return final_answer
