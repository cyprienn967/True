import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from classifier import HAPIClassifier  # Import hallucination classifier
import os
from huggingface_hub import login

# Authenticate to Hugging Face
login(token="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")

# Set model paths
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
CLASSIFIER_PATH = "./models/best_acc_model.pt"  # Ensure classifier is in models/
PROMPTS_FILE = "./prompts.txt"

# Output file paths
OUTPUT_DIR = "./outputs"
BASELINE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_output.txt")
FILTERED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "filtered_output.txt")
HALLUCINATIONS_LOG_FILE = os.path.join(OUTPUT_DIR, "flagged_hallucinations.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Llama-2 model and tokenizer
print(f"Loading Llama-2 model on {device}...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Load hallucination classifier
print("Loading Hallucination Classifier...")
classifier = HAPIClassifier(CLASSIFIER_PATH, device=device)

# Read prompts from file
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# Store results
baseline_results = []
filtered_results = []
flagged_hallucinations = []


def generate_text(prompt, max_length=400):
    """
    Generates text using the Llama-2 model.
    The max_length has been increased to produce roughly 200-300 words.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_filtered_text(prompt, classifier, max_length=400, temperature=1.0, top_k=50, max_attempts=5):
    """
    Generates text token-by-token with hallucination filtering.
    Each candidate token is sampled (using temperature and top-k filtering)
    and its hidden state is checked by the classifier. If it is flagged as a hallucination,
    the token is re-sampled (up to max_attempts) before accepting it.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated_ids = input_ids.clone()
    hallucinated_tokens = []

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
            # Apply temperature scaling
            logits = logits / temperature
            # Apply top-k filtering
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_values, dim=-1)

        valid_token_found = False
        chosen_token_id = None

        for attempt in range(max_attempts):
            sampled_index = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)
            # Extract scalar index from sampled_index to avoid extra dimensions
            sampled_index_scalar = sampled_index.item()
            token_id = top_k_indices[0, sampled_index_scalar].unsqueeze(0)  # shape: (1,)

            # Append token_id (reshaped to (1,1)) to the current sequence
            candidate_input_ids = torch.cat([generated_ids, token_id.unsqueeze(0)], dim=1)
            with torch.no_grad():
                candidate_outputs = model(candidate_input_ids, output_hidden_states=True)
                candidate_hidden_state = torch.cat(
                    (candidate_outputs.hidden_states[-1][:, -1, :],
                     candidate_outputs.hidden_states[-2][:, -1, :]),
                    dim=1
                )
            if classifier.classify(candidate_hidden_state):
                hallucinated_tokens.append(tokenizer.decode(token_id))
                # Try a new token candidate
                continue
            else:
                chosen_token_id = token_id
                valid_token_found = True
                break

        if not valid_token_found:
            # Default to the top candidate if all attempts are flagged
            chosen_token_id = top_k_indices[0, 0].unsqueeze(0)

        generated_ids = torch.cat([generated_ids, chosen_token_id.unsqueeze(0)], dim=1)

        if chosen_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, hallucinated_tokens


# Loop through prompts and generate outputs
for i, prompt in enumerate(prompts):
    try:
        print(f"({i+1}/{len(prompts)}) Generating for prompt: {prompt}")

        # Generate baseline output without hallucination filtering
        baseline_output = generate_text(prompt, max_length=400)

        # Generate filtered output (with hallucination detection)
        filtered_output, hallucinations = generate_filtered_text(prompt, classifier, max_length=400)

        # Store results
        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT:\n{baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT:\n{filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(f"PROMPT: {prompt}\nFLAGGED TOKENS: {', '.join(hallucinations)}\n")

    except Exception as e:
        print(f"⚠️ Error processing prompt: {prompt}\n{e}")

# Save outputs
with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(baseline_results))

with open(FILTERED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered_results))

with open(HALLUCINATIONS_LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(flagged_hallucinations))

print("\n✅ Generation complete! Check the outputs directory for results.")
