# generate_data.py

import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
import spacy
from utils.gen import chat_change_with_answer
from utils.model import get_model
import logging
import time  # just for optional sleep (to see the tqdm bar better)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_entities(text):
    """
    Extract entities from text using spaCy's NER.
    Returns a list of tuples (entity_text, start_char).
    """
    doc = nlp(text)
    entities = [(ent.text, ent.start_char) for ent in doc.ents]
    return entities

def chat_prompt(title):
    """
    Generate a simple prompt based on the title.
    For GPT-Neo, no chat formatting is needed.
    """
    return f"Tell me something about {title}."

def find_first_and_next_token(text, entity, idx, input_ids, p_=""):
    """
    Find the first and next tokens around the entity in the text.

    Args:
        text (str): The original text.
        entity (str): The entity text.
        idx (int): The starting character index of the entity.
        input_ids (list): Token IDs up to the entity.
        p_ (str): Additional prompt if any.

    Returns:
        tuple or None: (first_token, next_token, entity_length, last_id) or None if not found.
    """
    try:
        # Encode the entity
        entity_encoded = tokenizer.encode(entity, add_special_tokens=False)
        if not entity_encoded:
            return None
        first_token = input_ids[0][-1]  # Last token before the entity
        next_token = entity_encoded[0]  # First token of the entity
        entity_len = len(entity_encoded)
        last_id = input_ids[0][-1:]
        return first_token, next_token, entity_len, last_id
    except Exception as e:
        logging.error(f"Error in find_first_and_next_token: {e}")
        return None

logging.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')

logging.info("Logging into Hugging Face Hub...")
# Make sure your token is correct and hasn't expired.
# If you don't actually need to push to HF hub, you can comment the login out.
login(token="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")

# --------------------------------------------- #
model_type = "125M"  # GPT-Neo 1.3B
model_family = "gpt-neo"

wiki_path = "./auto-labeled/wiki"
output_path = f"./auto-labeled/output/{model_family}{model_type}"

topk_first_token = 4
windows = 2
topk_next_token = topk_first_token
# --------------------------------------------- #

logging.info(f"Loading model {model_family}{model_type}...")
model, tokenizer, generation_config, at_id = get_model(
    model_type, 
    model_family, 
    max_new_tokens=50
)

os.makedirs(output_path, exist_ok=True)

st = ""  # GPT-Neo does not require a special start token

data_types = ["train", "valid", "test"]
for data_type in data_types:
    logging.info(f"Processing data type: {data_type}")
    result = []

    # Expected JSON file name: wiki_train.json, wiki_valid.json, wiki_test.json
    data_file = f"{wiki_path}/wiki_{data_type}.json"
    if not os.path.exists(data_file):
        logging.warning(f"Data file {data_file} does not exist. Skipping.")
        continue

    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)

    # Log how many items we actually read
    logging.info(f"Loaded {len(data)} items from {data_file}")

    if len(data) == 0:
        logging.warning(f"No data found in {data_file}. Check the JSON content.")
        continue

    # Start iterating over data
    for ii, d in tqdm(enumerate(data), total=len(data), desc=f"Processing {data_type}"):
        # Optional short sleep to see the progress bar update in real time
        # time.sleep(0.01)

        # Check the keys we need
        if "title" not in d:
            logging.warning(f"No 'title' key for item index {ii}. Skipping.")
            continue
        if "sentences" not in d:
            logging.warning(f"No 'sentences' key for item index {ii}. Skipping.")
            continue

        # Some data might have empty sentences or only 1
        # The code uses the first 2 sentences if available
        if not d["sentences"]:
            logging.warning(f"Item index {ii} has an empty 'sentences' list. Skipping.")
            continue

        # Construct text
        # Safely handle case: if there's only 1 sentence, we just join that
        text = " ".join(d["sentences"][:2])

        entities_ = get_entities(text)
        if not entities_:
            logging.info(f"No named entities found in item index {ii}. Skipping.")
            # We still append a result for the doc, but empty "texts"
            ret = {
                "original_text": text,
                "title": d["title"],
                "texts": [],
                "new_entities": [],
                "original_entities": []
            }
            result.append(ret)
            continue

        # Remove duplicate offsets
        idx_set = set()
        entities = []
        for e_info in entities_:
            if e_info[1] not in idx_set:
                idx_set.add(e_info[1])
                entities.append(e_info)

        mytexts = []
        new_entities = []
        original_entity = []
        ret = {
            "original_text": text,
            "title": d["title"]
        }

        for (e, idx) in entities:
            # Skip if entity is at index 0 or if e is in the title
            if idx == 0 or (e in d["title"]):
                continue

            logging.debug(f"Processing entity: {e} at index: {idx}")

            # Tokenize up to that entity
            input_str = text[:idx].strip()
            if not input_str:
                continue

            input_id = tokenizer(input_str, return_tensors='pt')
            input_ids = input_id["input_ids"].to(model.device)
            attention_mask = input_id["attention_mask"].to(model.device)

            tokens = find_first_and_next_token(text, e, idx, input_ids.tolist(), p_="")
            if not tokens:
                logging.warning(f"Tokens not found for entity: {e} (index {idx}). Skipping.")
                continue
            first_, next_, entity_len, last_id = tokens

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )

            # Check "scores"
            if not hasattr(output, "scores") or not output.scores:
                logging.warning(f"Model output does not have 'scores'. Skipping token filtering.")
                continue

            # The last token's scores
            scores = output.scores[-1]
            _, indices = torch.topk(scores, k=topk_first_token)
            if first_ in indices.tolist():
                logging.debug(f"First token {first_} found in top {topk_first_token}. Skipping entity.")
                continue

            sequences = output.sequences
            # Expand generation to check next_ token in topk
            for _ in range(entity_len + windows):
                with torch.no_grad():
                    new_seq_len = sequences.shape[1]
                    new_attention_mask = torch.ones(
                        (sequences.shape[0], new_seq_len),
                        dtype=torch.long
                    ).to(model.device)

                    output = model.generate(
                        input_ids=sequences,
                        attention_mask=new_attention_mask,
                        **generation_config
                    )
                if not hasattr(output, "scores") or not output.scores:
                    logging.warning("No 'scores' during window generation.")
                    break

                scores = output.scores[-1]
                _, indices = torch.topk(scores, k=topk_next_token)
                if next_ in indices.tolist():
                    logging.debug(f"Next token {next_} found in top {topk_next_token}. Breaking out.")
                    break
                sequences = output.sequences

            if next_ not in indices.tolist():
                logging.debug(f"Next token {next_} not found in top {topk_next_token} after expansion. Skipping.")
                continue

            new_sequence = sequences[0].tolist()
            new_entity_id = new_sequence[len(input_ids[0]):]

            # Construct the new text
            if model_family == "falcon":
                all_new_text_id = (
                    input_ids[0].tolist() +
                    [204, 43, 204] +
                    new_entity_id +
                    [204, 43, 204] +
                    last_id
                )
            elif isinstance(at_id, list):
                all_new_text_id = (
                    input_ids[0].tolist() +
                    [at_id[0]] +
                    new_entity_id +
                    [at_id[0]] +
                    last_id
                )
            elif at_id is not None:
                all_new_text_id = (
                    input_ids[0].tolist() +
                    [at_id] +
                    new_entity_id +
                    [at_id] +
                    last_id
                )
            else:
                # For GPT-Neo, just concatenate
                all_new_text_id = input_ids[0].tolist() + new_entity_id + last_id

            mytext = tokenizer.decode(all_new_text_id, skip_special_tokens=True)
            mytext = mytext.replace("<s>", "").replace("</s>", "")

            # Attempt to extract entity between '@' symbols
            at_start = mytext.find("@")
            at_end = mytext.rfind("@")
            if at_start == -1 or at_end == -1 or at_start == at_end:
                logging.warning(f"No valid '@' found in generated text: {mytext}")
                continue

            new_entity_str = mytext[at_start + 1 : at_end].strip().lower()

            # If new_entity_str is basically in the original text, skip
            if any(
                (ee.strip() in text.lower()) 
                for ee in new_entity_str.split(" ")
            ) or (e.lower() in new_entity_str):
                logging.debug(f"New entity '{new_entity_str}' found in original text or same as {e}. Skipping.")
                continue

            mytexts.append(mytext)
            new_entities.append(new_entity_str)
            original_entity.append((e, idx))

        # Save in "ret"
        ret["texts"] = mytexts
        ret["new_entities"] = new_entities
        ret["original_entities"] = original_entity
        result.append(ret)

    # Save results for this data_type
    output_file = f"{output_path}/data_{data_type}.json"
    with open(output_file, "w+", encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    logging.info(f"Saved processed data to {output_file}")
