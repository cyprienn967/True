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

# Load Llama-2 model and tokenizer
print(f"Loading Llama-2 model on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Load hallucination classifier
print("Loading Hallucination Classifier...")
classifier = HAPIClassifier(CLASSIFIER_PATH, device="cuda" if torch.cuda.is_available() else "cpu")

# Read prompts from file
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# Store results
baseline_results = []
filtered_results = []
flagged_hallucinations = []


def generate_text(prompt, max_length=100):
    """Generates text using Llama-2 model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_hidden_states(input_ids):
    """Extracts and concatenates the last two hidden state layers to match the classifier's 8192-dim input."""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All hidden states

        # Extract last two layers and concatenate them
        last_layer = hidden_states[-1][:, -1, :]  # Shape: (1, 4096)
        second_last_layer = hidden_states[-2][:, -1, :]  # Shape: (1, 4096)
        
        concatenated_hidden_states = torch.cat((last_layer, second_last_layer), dim=1)  # Shape: (1, 8192)
        return concatenated_hidden_states


def generate_with_hallucination_filtering(prompt, classifier, max_length=100):
    """Generates text token-by-token, detecting hallucinations and regenerating flagged tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    output_ids = model.generate(**inputs, max_length=max_length, output_scores=True, return_dict_in_generate=True)
    decoded_output = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)

    hallucinations = []
    tokens = decoded_output.split()
    filtered_tokens = []

    for token in tokens:
        token_tensor = tokenizer(token, return_tensors="pt").input_ids.to(model.device)
        hidden_state = extract_hidden_states(token_tensor)

        is_hallucination = classifier.classify(hidden_state)  # Check hallucination

        if is_hallucination:
            hallucinations.append(token)
            new_output_ids = model.generate(token_tensor, max_new_tokens=1)  # ✅ FIXED HERE
            new_token = tokenizer.decode(new_output_ids[0], skip_special_tokens=True)
            filtered_tokens.append(new_token)
        else:
            filtered_tokens.append(token)

    return " ".join(filtered_tokens), hallucinations


# Loop through prompts
for i, prompt in enumerate(prompts):
    try:
        print(f"({i+1}/{len(prompts)}) Generating for prompt: {prompt}")

        # Generate baseline output
        baseline_output = generate_text(prompt, max_length=100)

        # Generate filtered output (with hallucination detection)
        filtered_output, hallucinations = generate_with_hallucination_filtering(prompt, classifier, max_length=100)

        # Store results
        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT: {baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT: {filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(f"PROMPT: {prompt}\nFLAGGED TOKENS: {', '.join(hallucinations)}\n")

    except Exception as e:
        print(f"⚠️ Error processing prompt: {prompt}\n{e}")

# Save outputs
with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(baseline_results))

with open(FILTERED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(filtered_results))

with open(HALLUCINATIONS_LOG_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(flagged_hallucinations))

print("\n✅ Generation complete! Check the outputs directory for results.")
