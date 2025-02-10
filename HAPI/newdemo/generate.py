import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from classifier import HAPIClassifier  # Import the hallucination classifier
import os

# Set paths
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
CLASSIFIER_PATH = "./models/best_acc_model.pt"  # Ensure classifier is in models/
PROMPTS_FILE = "./prompts.txt"

# Output files
BASELINE_OUTPUT_FILE = "./outputs/baseline_output.txt"
FILTERED_OUTPUT_FILE = "./outputs/filtered_output.txt"
HALLUCINATIONS_LOG_FILE = "./outputs/flagged_hallucinations.txt"

# Load Llama-2 model and tokenizer
print("Loading Llama-2 model...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Load hallucination classifier
print("Loading Hallucination Classifier...")
classifier = HAPIClassifier(CLASSIFIER_PATH, device="cuda" if torch.cuda.is_available() else "cpu", input_size=4096 * 2)

# Read prompts from file
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]

# Store results
baseline_results = []
filtered_results = []
flagged_hallucinations = []

# Define generation function
def generate_text(prompt, max_length=100):
    """Generates text using the Llama model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Define function with hallucination filtering
def generate_token_by_token(prompt, classifier, max_length=100):
    """Generates text token-by-token, flagging hallucinations and regenerating if needed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length, output_scores=True, return_dict_in_generate=True)

    decoded_output = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
    hallucinations = []

    tokens = decoded_output.split()  # Split into words for token-by-token processing
    filtered_tokens = []

    for token in tokens:
        token_tensor = torch.tensor(tokenizer.encode(token, return_tensors="pt")).to(model.device)
        is_hallucination = classifier.classify(token_tensor)  # Check hallucination

        if is_hallucination:
            hallucinations.append(token)
            new_output_ids = model.generate(token_tensor, max_length=1)  # Regenerate token
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
        
        # Generate filtered output (with hallucination filtering)
        filtered_output, hallucinations = generate_token_by_token(prompt, classifier, max_length=100)

        # Store results
        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT: {baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT: {filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(f"PROMPT: {prompt}\nFLAGGED TOKENS: {', '.join(hallucinations)}\n")

    except Exception as e:
        print(f"⚠️ Error processing prompt: {prompt}\n{e}")

# Ensure output directory exists
os.makedirs("./outputs", exist_ok=True)

# Save outputs
with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(baseline_results))

with open(FILTERED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(filtered_results))

with open(HALLUCINATIONS_LOG_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(flagged_hallucinations))

print("\n✅ Generation complete! Check the outputs directory for results.")
