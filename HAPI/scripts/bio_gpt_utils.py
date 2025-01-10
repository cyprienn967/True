import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class BioGPTGenerator:
    def __init__(self, model_name="microsoft/BioGPT-Large", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[BioGPT] Loading model {model_name} on device={self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def clean_model_output(self, text: str) -> str:
        """
        Remove special tokens and extraneous HTML-like placeholders from the final text.
        """
        # Remove any `< ... >` patterns
        text = re.sub(r"<[^>]+>", "", text)
        # If "Answer:" remains, split on it
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        # You could remove remaining non-ASCII, etc.
        text = text.strip()
        return text

    def generate_raw_answer(self, question: str, max_length: int = 200) -> str:
        prompt = (
            f"Question: {question}\n"
            "You are a helpful assistant. Provide a concise, direct, and relevant answer.\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.3,        # Lower temp => more deterministic
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.clean_model_output(answer)
        return answer

    def truncate_context(self, context: str, max_tokens: int) -> str:
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
        Use retrieved docs as context and generate an answer. 
        We strongly encourage the model to produce a short, direct answer.
        """
        # Combine retrieved chunks into a single context block
        context_block = "\n".join([f"- {doc}" for doc in relevant_docs])

        # Truncate the combined context to ensure it fits within the token limit
        max_context_tokens = 384  # somewhat smaller context to reduce overhead
        context_block = self.truncate_context(context_block, max_context_tokens)

        # More explicit instructions to produce a short, final answer
        prompt = (
            "You are a helpful biomedical question-answering system. Below is relevant context:\n"
            f"{context_block}\n\n"
            "Please read the context and the question carefully, then produce a concise, direct answer. "
            "Do not include any extraneous text or special tokens. If uncertain, answer concisely. "
            "Question: "
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
        answer = self.clean_model_output(answer)
        return answer
