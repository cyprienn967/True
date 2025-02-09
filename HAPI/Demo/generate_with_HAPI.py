# generate_with_hapi.py
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests
import numpy as np

# Change this URL to the actual URL of your HAPI server
HAPI_URL = "http://localhost:5000/hapi"

def extract_features(output):
    """
    Extract features from the generation output.
    We assume that output.hidden_states is a list of all layer hidden states.
    For demonstration, we take:
      - the last token hidden state of the last layer
      - the mean of all token hidden states in the last layer
    and concatenate them.
    """
    # output.hidden_states[-1]: shape [batch_size, seq_len, hidden_dim]
    last_layer = output.hidden_states[-1]
    # Extract last token hidden state
    last_token = last_layer[0, -1, :]  # shape: [hidden_dim]
    # Compute mean of the last layer (over tokens)
    last_mean = torch.mean(last_layer[0], dim=0)
    # Convert both to lists and concatenate
    features = last_token.detach().cpu().numpy().tolist() + last_mean.detach().cpu().numpy().tolist()
    return features

def call_hapi(features):
    """
    Call the HAPI endpoint with a JSON payload containing the features.
    Expect a response like {"score": float}. If the score exceeds threshold,
    it indicates a hallucination.
    """
    payload = {"features": features}
    try:
        response = requests.post(HAPI_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("score", 0.0)
    except Exception as e:
        print("Error calling HAPI:", e)
        # On error, assume token is not hallucinated
        return 0.0

def generate_with_hapi(prompt, model, tokenizer, max_new_tokens=50, threshold=0.5):
    """
    Generate text token-by-token, checking each generated token with HAPI.
    If HAPI flags a token (score > threshold), that token is discarded and a new one is generated.
    """
    # Tokenize the prompt
    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # We'll iterate for max_new_tokens steps
    for i in range(max_new_tokens):
        # Generate one token at a time
        output = model.generate(generated,
                                max_new_tokens=1,
                                do_sample=True,
                                temperature=0.7,
                                return_dict_in_generate=True,
                                output_hidden_states=True)
        
        # Extract features from the hidden states
        features = extract_features(output)
        # Call HAPI to get a hallucination score for this token
        score = call_hapi(features)
        
        # If score is above threshold, we discard this token and try again.
        # (Optionally, you can limit the number of retries per token.)
        if score > threshold:
            print(f"Step {i+1}: Hallucination detected (score {score:.2f}). Regenerating token.")
            continue
        else:
            # Accept the token by updating the generated sequence
            generated = output.sequences
        
    # Decode the final sequence to text
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text

def main():
    model_name = "llamachat7b"  # Adjust if needed
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model.to("cuda:0")
    
    prompt = "The story begins with a young adventurer who"
    
    print("Generating text WITHOUT HAPI:")
    # Simple generation without hallucination filtering
    output = model.generate(tokenizer(prompt, return_tensors="pt").input_ids.to(model.device),
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.7)
    text_no_hapi = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text_no_hapi)
    
    print("\nGenerating text WITH HAPI (Hallucination API):")
    text_with_hapi = generate_with_hapi(prompt, model, tokenizer, max_new_tokens=50, threshold=0.5)
    print(text_with_hapi)

if __name__ == "__main__":
    main()
