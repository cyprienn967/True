import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class LlamaChat7B:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_token_by_token(self, prompt, classifier, max_length=100):
        """
        Generates text one token at a time, checking after each token
        if its hidden state is flagged by the hallucination classifier.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.clone()
        hallucinations = []

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                # Extract hidden states from the last two layers for the last token.
                last_layer = outputs.hidden_states[-1][:, -1, :]
                second_last_layer = outputs.hidden_states[-2][:, -1, :]
                token_hidden_state = torch.cat((last_layer, second_last_layer), dim=1)  # Shape: (1, 8192)

            if classifier.classify(token_hidden_state):  # Check if the token is hallucinated
                hallucinations.append(self.tokenizer.decode(next_token_id[0]))
                # Skip adding this token and try generating a replacement in the next iteration.
                continue

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            input_ids = generated_ids  # Update input for the next token

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True), hallucinations
