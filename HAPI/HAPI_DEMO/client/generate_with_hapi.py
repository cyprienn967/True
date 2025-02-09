# client/generate_with_hapi.py
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests
import numpy as np
import os
from huggingface_hub import login
login(token="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")


# URL of the HAPI server
HAPI_URL = "http://localhost:5000/hapi"

# Path to the log file (relative to the project root)
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "hallucination_log.txt")

def log_flagged(token_text, score, prompt, attempt):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Flagged Token: '{token_text}', Score: {score:.4f}, Prompt: '{prompt}', Attempt: {attempt}\n")

def extract_features(output):
    """
    Extract features from output.hidden_states:
      - Last token hidden state (from the last layer)
      - Mean of the last layer hidden states
    Concatenate these to form a single feature vector.
    """
    last_layer = output.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
    last_token = last_layer[0, -1, :]       # shape: [hidden_dim]
    last_mean = torch.mean(last_layer[0], dim=0)
    features = last_token.detach().cpu().numpy().tolist() + last_mean.detach().cpu().numpy().tolist()
    return features

def call_hapi(features):
    payload = {"features": features}
    try:
        response = requests.post(HAPI_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("score", 0.0)
    except Exception as e:
        print("Error calling HAPI:", e)
        return 0.0

def generate_with_hapi(prompt, model, tokenizer, max_new_tokens=50, threshold=0.5, max_attempts=5):
    # Start with the prompt tokens.
    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    accepted_sequence = generated
    for token_idx in range(max_new_tokens):
        attempt = 0
        while attempt < max_attempts:
            output = model.generate(accepted_sequence,
                                    max_new_tokens=1,
                                    do_sample=True,
                                    temperature=0.7,
                                    return_dict_in_generate=True,
                                    output_hidden_states=True)
            # Get the generated token id and decode it.
            new_token_id = output.sequences[0, -1].item()
            new_token = tokenizer.decode([new_token_id], skip_special_tokens=True)
            features = extract_features(output)
            score = call_hapi(features)
            if score > threshold:
                # Log the flagged token.
                log_flagged(new_token, score, prompt, attempt+1)
                print(f"Token '{new_token}' flagged (score {score:.4f}), attempt {attempt+1}")
                attempt += 1
            else:
                accepted_sequence = output.sequences
                break
        if attempt == max_attempts:
            print(f"Max attempts reached for token index {token_idx}. Accepting token '{new_token}' despite high score {score:.4f}.")
            accepted_sequence = output.sequences
    generated_text = tokenizer.decode(accepted_sequence[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name = "llamachat7b"  # Adjust if needed
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Read prompts
    with open("prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    output_file = "generation_with_hapi.txt"
    with open(output_file, "w", encoding="utf-8") as out_f:
        for prompt in prompts:
            generated = generate_with_hapi(prompt, model, tokenizer, max_new_tokens=50, threshold=0.5)
            out_f.write(f"Prompt: {prompt}\nGenerated: {generated}\n{'-'*40}\n")
            print(f"Generated text for prompt (with HAPI): {prompt}")
    
    print(f"Generation with HAPI saved to {output_file}")

if __name__ == "__main__":
    main()
