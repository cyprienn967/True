# File: HAPI/scripts2/api_server.py

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import re
import warnings
from mlp_detection import HallucinationDetector  # Ensure this import path is correct

app = Flask(__name__)

# ------------------- Define Cleaning Functions -------------------

def clean_text(text: str) -> str:
    """
    Removes enumerations, references, placeholders, and excess whitespace.
    """
    # Remove patterns like (1), (2), etc.
    text = re.sub(r'\(\d+\)', '', text)
    
    # Remove patterns like [1], [PMID: XXXX], etc.
    text = re.sub(r'\[\w+[: ]*\d+\]', '', text)
    
    # Remove multiple periods and other placeholders
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove trailing incomplete sentences ending with dots
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'\.\.\.$', '', text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_unwanted_phrases(text: str) -> str:
    """
    Removes specific unwanted phrases from the text.
    """
    unwanted_phrases = [
        "The answer to this question is",
        "This is a",
        "In this study",
        "We think that",
        "This paper",
        "We believe that",
        "KEY WORDS:",
        "Key words:",
        "KEY POINTS:",
        "This conundrum",
        "A systematic review of",
        "A commentary by",
        "A case study in",
        "BACKGROUND AND OBJECTIVES:",
        "METHODS / MATERIALS:",
        "CONCLUSIONS:",
        "RESULTS:",
        "INTRODUCTION:",
        "DISCUSSION:",
        "ABSTRACT TRUNCATED AT",
        "REVIEWERS:",
        "IMPLICATIONS FOR",
        "Some weights of RobertaModel were not initialized",
        "You should probably TRAIN this model",
        # Add more phrases as needed
    ]
    
    for phrase in unwanted_phrases:
        # Use regex for case-insensitive replacement and to remove trailing spaces
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    
    return text

def truncate_incomplete_sentences(text: str) -> str:
    """
    Truncates the text to remove incomplete sentences at the end.
    """
    # Split the text into sentences based on punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Keep only complete sentences
    complete_sentences = [s for s in sentences if re.match(r'.*[.!?]$', s)]
    
    return ' '.join(complete_sentences)

def normalize_whitespace_punctuation(text: str) -> str:
    """
    Normalizes whitespace and ensures proper spacing after punctuation.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    
    # Trim leading and trailing spaces
    text = text.strip()
    
    return text

def comprehensive_cleaning(text: str) -> str:
    """
    Applies all cleaning steps to the text.
    """
    text = clean_text(text)
    text = truncate_incomplete_sentences(text)
    text = normalize_whitespace_punctuation(text)
    text = remove_unwanted_phrases(text)
    return text

# ------------------- Initialize Models -------------------

# Load BioGPT model and tokenizer
MODEL_NAME = "microsoft/BioGPT"  # Replace with the exact BioGPT model you're using
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Initialize the Hallucination Detector
detector = HallucinationDetector(
    mlp_path="../data/best_mlp_model.pt",
    device="cpu",  # Change to "cuda" if GPU is available and desired
    model_name=MODEL_NAME,
    hidden_dim=1024  # Adjust based on BioGPT's hidden size (check model.config.hidden_size)
)

# ------------------- Define Answer Generation Function -------------------

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a rank greater than top_k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Sort the logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the mask right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the mask to the original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def generate_answer(question: str, max_length: int = 200, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.5, hallucination_threshold: float = 0.5):
    """
    Generates an answer to the given question using BioGPT.
    At each generation step, checks for hallucinations using the MLP model.
    If hallucination is detected, it skips appending the hallucinated token.
    """
    # Encode the question
    input_ids = tokenizer.encode(question, return_tensors='pt').to(detector.device)
    
    # Initialize generated tokens with input_ids
    generated_ids = input_ids
    max_new_tokens = max_length
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        # Get the model's output
        with torch.no_grad():
            outputs = model(generated_ids, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states[-4:]  # Last 4 hidden layers
        
            # Extract embeddings
            last_token_emb = outputs.hidden_states[-1][:, -1, :]   # [1, hidden_dim]
            mean_last4_emb = torch.mean(torch.stack(hidden_states, dim=0), dim=0)[:, -1, :]  # [1, hidden_dim]
            combined_emb = torch.cat([last_token_emb, mean_last4_emb], dim=1)  # [1, 2*hidden_dim]
            print(f"Combined Embedding Shape: {combined_emb.shape}")  # Debugging
        
        # Apply temperature, top_p, top_k, and repetition_penalty
        next_token_logits = next_token_logits / temperature
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_value=-float('Inf'))
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        
        # Sample the next token
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # Predict hallucination using the MLP
        logits = detector.mlp(combined_emb)  # [1, 2]
        probs = torch.softmax(logits, dim=1)  # [1, 2]
        hallucination_prob = probs[0, 1].item()  # Probability of class 1 (hallucination)
        print(f"Hallucination Probability: {hallucination_prob}")  # Debugging
        
        if hallucination_prob > hallucination_threshold:
            # Hallucination detected, regenerate this token
            print("[INFO] Hallucination detected. Skipping token.")
            continue  # Skip appending the hallucinated token
        else:
            # Append the token to the generated_ids
            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
        
        # Check for end of generation
        if next_token.item() == eos_token_id:
            break
    
    # Decode the generated tokens to text
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return answer

# ------------------- Define API Endpoint -------------------

@app.route('/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate answers using BioGPT with optional hallucination detection.
    Expects a JSON payload with a 'question' field and an optional 'with_hallucination_filter' boolean.
    Returns a JSON response with the 'final_answer' field.
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid input. "question" field is required.'}), 400
    
    question = data['question']
    with_hallucination_filter = data.get('with_hallucination_filter', False)
    
    try:
        if with_hallucination_filter:
            answer = generate_answer(question, hallucination_threshold=0.5)
        else:
            # Generate without hallucination detection (RAW)
            answer = generate_answer(question, hallucination_threshold=0.0)  # Effectively disabling the filter
        # Apply comprehensive cleaning
        answer_clean = comprehensive_cleaning(answer)
        return jsonify({'final_answer': answer_clean}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------- Run the Flask App -------------------

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000)
