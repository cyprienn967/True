import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class BioMedLMGenerator:
    def __init__(self, model_name="StanfordBioMedLM/BioMedLM", device=None):
        """
        Initialize the BioMedLM 2.7B model from Hugging Face.
        Adjust model_name if needed to match the correct checkpoint path on HF.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[BioMedLM] Loading model {model_name} on device={self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def clean_model_output(self, text: str) -> str:
        """
        Remove special tokens or placeholders from final text.
        """
        text = re.sub(r"<[^>]+>", "", text)
        # If "Answer:" remains, optionally split on it
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        return text.strip()

    def generate_raw_answer(self, question: str, max_length: int = 200) -> str:
        """
        Generate a direct answer using BioMedLM.
        """
        prompt = (
            f"{question}\n"
            "You are a helpful biomedical assistant. Provide a concise, direct answer.\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_model_output(answer)

    def truncate_context(self, context: str, max_tokens: int) -> str:
        """
        Token-level truncation to limit context size.
        """
        tokenized_context = self.tokenizer(
            context,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
            padding=False
        )
        truncated_context = self.tokenizer.decode(
            tokenized_context["input_ids"][0],
            skip_special_tokens=True
        )
        return truncated_context

    def generate_rag_answer(self, question: str, relevant_docs: list[str], max_new_tokens: int = 80) -> str:
        """
        Use retrieved docs as context to produce a retrieval-augmented (RAG) style answer.
        """
        # Combine retrieved chunks into a single context block
        context_block = "\n".join([f"- {doc}" for doc in relevant_docs])

        # Truncate the combined context so the prompt isn't too large
        max_context_tokens = 384
        context_block = self.truncate_context(context_block, max_context_tokens)

        prompt = (
            "You are a helpful biomedical question-answering system. Below is relevant context:\n"
            f"{context_block}\n\n"
            "Read the context and the question carefully, then produce a concise, direct answer.\n"
            f"{question}\n"
            "Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_model_output(answer)
