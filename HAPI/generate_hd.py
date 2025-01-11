# generate_hd.py

import os
import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer
from tqdm import tqdm
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------- #
# For GPT-Neo, you can enable CUDA if available
# Otherwise, set to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA if not using GPU
model_type = "1.3B"  # GPT-Neo 1.3B
model_family = "gpt-neo"
result_path = f"./auto-labeled/output/{model_family}{model_type}"
# --------------------------------------------- #

logging.info(f"Model Family and Type: {model_family}{model_type}")

# Load the model and tokenizer
model, tokenizer, generation_config, at_id = get_model(model_type, model_family, max_new_tokens=50)
model.to("cpu")  # Ensure the model runs on CPU

# Define prompt_chat function (GPT-Neo is not chat-based)
def prompt_chat(title):
    return f"Tell me something about {title}."

def get_tokenized_ids(otext, title=None):
    """
    Tokenize the text for hidden state extraction.
    """
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    encoded = tokenizer(text.strip(), return_tensors='pt')
    # For GPT-Neo, no chat formatting needed
    return encoded['input_ids'].tolist()

def get_hd(text, title=None):
    """
    Generate hidden states for the given text.

    Args:
        text (str): The input text.
        title (str, optional): The title context. Defaults to None.

    Returns:
        tuple: (hidden_states, mean1, mean2)
    """
    ids = get_tokenized_ids(text, title)
    # Ensure the model runs on the CPU
    with torch.no_grad():
        outputs = model(torch.tensor(ids).to("cpu"), output_hidden_states=True)
    hd = outputs.hidden_states  # Tuple of hidden states
    # Compute the mean of the last hidden state
    hds_last = hd[-1][0].clone().detach()  # Last layer
    hds_mean_1 = torch.mean(hds_last, dim=0).tolist()
    hds_mean_2 = torch.mean(hds_last, dim=0).tolist()

    return hd, hds_mean_1, hds_mean_2

# Processing loop
for data_type in ["train", "valid", "test"]:
    logging.info(f"Generating hidden states for data type: {data_type}")
    data_file = f"{result_path}/data_{data_type}.json"
    if not os.path.exists(data_file):
        logging.warning(f"Data file {data_file} does not exist. Skipping.")
        continue

    data = json.load(open(data_file, encoding='utf-8'))
    results_last = []
    results_mean1 = []
    results_mean2 = []

    for k in tqdm(data, desc=f"Processing {data_type}"):
        hd_last = []
        hd_mean1 = []
        hd_mean2 = []

        hdl_origin, hdm1_origin, hdm2_origin = get_hd(k["original_text"], k["title"])

        for t in k["texts"]:
            hdl, hdm1, hdm2 = get_hd(t, k["title"])
            hd_last.append(hdl[-1][0].tolist())  # Last hidden state
            hd_mean1.append(hdm1)
            hd_mean2.append(hdm2)

        results_last.append({
            "right": hdl_origin[-1][0].tolist(),
            "hallu": hd_last,
        })
        results_mean1.append({
            "right": hdm1_origin,
            "hallu": hd_mean1,
        })
        results_mean2.append({
            "right": hdm2_origin,
            "hallu": hd_mean2,
        })

    # Save results
    output_file_last = f"{result_path}/last_token_mean_{data_type}.json"
    with open(output_file_last, "w+", encoding='utf-8') as f:
        json.dump(results_last, f)
    logging.info(f"Saved last hidden states to {output_file_last}")

    output_file_mean1 = f"{result_path}/last_mean1_{data_type}.json"
    with open(output_file_mean1, "w+", encoding='utf-8') as f:
        json.dump(results_mean1, f)
    logging.info(f"Saved mean1 hidden states to {output_file_mean1}")

    output_file_mean2 = f"{result_path}/last_mean2_{data_type}.json"
    with open(output_file_mean2, "w+", encoding='utf-8') as f:
        json.dump(results_mean2, f)
    logging.info(f"Saved mean2 hidden states to {output_file_mean2}")
