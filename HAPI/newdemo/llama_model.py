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
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.clone()
        hallucinations = []

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                # Get hidden states (assuming using `last_hidden_state` layer output)
                hidden_states = outputs.logits[:, -1, :].cpu().numpy().flatten()  

            if classifier.predict(hidden_states):  # Check if hallucinated
                hallucinations.append(self.tokenizer.decode(next_token_id[0]))
                continue  # Skip adding to output, regenerate next iteration

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            input_ids = generated_ids  # Update input for next token

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True), hallucinations
