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


def extract_hidden_states(input_ids):
    """
    Extracts and concatenates the last two hidden state layers (for the final token)
    to produce the feature vector for the classifier.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All hidden states

        # Extract the last token's representation from the last two layers
        last_layer = hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
        second_last_layer = hidden_states[-2][:, -1, :]  # (batch_size, hidden_dim)
        
        concatenated_hidden_states = torch.cat((last_layer, second_last_layer), dim=1)  # (batch_size, 8192)
        return concatenated_hidden_states


def generate_sentence(context, sentence_max_length=60):
    """
    Generates a sentence using the current context as prompt.
    It uses a relatively short maximum length to keep generation sentence-bound.
    """
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=sentence_max_length, do_sample=True, top_k=50)
    sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the first sentence if multiple are returned
    if ". " in sentence:
        sentence = sentence.split(". ")[0] + "."
    return sentence.strip()


def sentence_contains_hallucination(sentence, classifier):
    """
    Checks a sentence token-by-token. Returns True if any token is flagged by the classifier.
    """
    tokens = sentence.split()
    for token in tokens:
        token_tensor = tokenizer(token, return_tensors="pt").input_ids.to(model.device)
        hidden_state = extract_hidden_states(token_tensor)
        if classifier.classify(hidden_state):
            return True
    return False


def generate_with_hallucination_filtering(prompt, classifier, desired_word_count=250, sentence_max_length=60, max_regen_attempts=3):
    """
    Generates text one sentence at a time, checking each sentence for hallucinations as soon as it is generated.
    Each new sentence is generated with the updated context (original prompt plus previously generated valid sentences).
    
    Parameters:
      - prompt: the input prompt.
      - classifier: the hallucination classifier.
      - desired_word_count: the target word count for the generated passage.
      - sentence_max_length: maximum token length for each sentence.
      - max_regen_attempts: maximum number of regeneration attempts per sentence if hallucinations are detected.
    """
    context = prompt
    generated_sentences = []
    hallucinated_sentences = []

    # Continue generating sentences until the desired word count is reached.
    while len(" ".join(generated_sentences).split()) < desired_word_count:
        sentence = generate_sentence(context, sentence_max_length=sentence_max_length)
        attempts = 0

        # Check the generated sentence for hallucinations.
        while sentence_contains_hallucination(sentence, classifier) and attempts < max_regen_attempts:
            print(f"🔴 Hallucination detected in sentence: '{sentence}'. Regenerating sentence (attempt {attempts+1})...")
            hallucinated_sentences.append(sentence)
            sentence = generate_sentence(context, sentence_max_length=sentence_max_length)
            attempts += 1

        # Append the (hopefully) valid sentence and update context.
        generated_sentences.append(sentence)
        context += " " + sentence

        # If generation stalls (empty sentence), break out.
        if not sentence.strip():
            break

    generated_text = " ".join(generated_sentences)
    return generated_text, hallucinated_sentences


def generate_full_text(prompt, max_length=400):
    """
    Generates a baseline passage without sentence-level filtering.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Loop through prompts and generate outputs
for i, prompt in enumerate(prompts):
    try:
        print(f"({i+1}/{len(prompts)}) Generating for prompt: {prompt}")

        # Generate baseline output without hallucination filtering
        baseline_output = generate_full_text(prompt, max_length=400)

        # Generate filtered output (sentence-by-sentence hallucination filtering)
        filtered_output, hallucinations = generate_with_hallucination_filtering(prompt, classifier, desired_word_count=250, sentence_max_length=60)

        baseline_results.append(f"PROMPT: {prompt}\nBASELINE OUTPUT:\n{baseline_output}\n")
        filtered_results.append(f"PROMPT: {prompt}\nFILTERED OUTPUT:\n{filtered_output}\n")

        if hallucinations:
            flagged_hallucinations.append(f"PROMPT: {prompt}\nFLAGGED SENTENCES: {', '.join(hallucinations)}\n")

    except Exception as e:
        print(f"⚠️ Error processing prompt: {prompt}\n{e}")

# Save outputs to disk
with open(BASELINE_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(baseline_results))

with open(FILTERED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered_results))

with open(HALLUCINATIONS_LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(flagged_hallucinations))

print("\n✅ Generation complete! Check the outputs directory for results.")
