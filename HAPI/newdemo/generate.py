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


def generate_text(prompt, max_length=100):
    """Generates text using Llama-2 model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def extract_hidden_states(input_ids):
    """
    Extracts and concatenates the last two hidden state layers to match the classifier's 8192-dim input.
    Note: The model is in fp16, so the resulting tensor is cast to float32 in the classifier.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All hidden states

        # Extract the last two layers and concatenate the final token's representations.
        last_layer = hidden_states[-1][:, -1, :]  # Shape: (batch_size, hidden_dim)
        second_last_layer = hidden_states[-2][:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        concatenated_hidden_states = torch.cat((last_layer, second_last_layer), dim=1)  # Shape: (batch_size, 8192)
        return concatenated_hidden_states


def generate_with_hallucination_filtering(prompt, classifier, max_length=150):
    """
    Generates text sentence-by-sentence, detecting hallucinations at token-level.
    If any token in a sentence is flagged as hallucinated, the full sentence is regenerated.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length, output_scores=True, return_dict_in_generate=True)

    decoded_output = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
    sentences = decoded_output.split(". ")  # Split into sentences

    filtered_sentences = []
    hallucinations = []

    for sentence in sentences:
        tokens = sentence.split()
        flag_hallucination = False  # Track if any token is flagged

        for token in tokens:
            token_tensor = tokenizer(token, return_tensors="pt").input_ids.to(model.device)
            hidden_state = extract_hidden_states(token_tensor)

            if classifier.classify(hidden_state):  # If a single token is classified as hallucinated
                flag_hallucination = True
                hallucinations.append(sentence)  # Store the entire sentence
                break  # Stop checking this sentence

        if flag_hallucination:
            print(f"🔴 Hallucination detected! Regenerating: {sentence}")
            sentence_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            new_output_ids = model.generate(sentence_tensor, max_length=30)  # Regenerate the full sentence
            new_sentence = tokenizer.decode(new_output_ids[0], skip_special_tokens=True)
            filtered_sentences.append(new_sentence)
        else:
            filtered_sentences.append(sentence)

    return ". ".join(filtered_sentences), hallucinations


# Loop through prompts and generate outputs
for i, prompt in enumerate(prompts):
    try:
        print(f"({i+1}/{len(prompts)}) Generating for prompt: {prompt}")

        # Generate baseline output
        baseline_output = generate_text(prompt, max_length=150)

        # Generate filtered output (with hallucination detection)
        filtered_output, hallucinations = generate_with_hallucination_filtering(prompt, classifier, max_length=150)

        # Store results
        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT: {baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT: {filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(f"PROMPT: {prompt}\nFLAGGED SENTENCES: {', '.join(hallucinations)}\n")

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
