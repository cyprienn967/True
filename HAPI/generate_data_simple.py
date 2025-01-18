#!/usr/bin/env python3
"""
generate_data_simple.py

This script reads wiki_{train,valid,test}.json files in ./auto-labeled/wiki/
and produces data_{train,valid,test}.json in ./auto-labeled/output/gpt-neo125M/
just like the original generate_data.py, but uses a simple random "real entity" replacement
instead of repeated calls to model.generate(...) or placeholders.
"""

import os
import json
import spacy
import logging
import random
from tqdm import tqdm

##############################################################################
# Configuration
##############################################################################
model_type = "125M"            # e.g., GPT-Neo 125M
model_family = "gpt-neo"       # used for naming output directory
wiki_path = "./auto-labeled/wiki"
output_path = f"./auto-labeled/output/{model_family}{model_type}"

# A large list of possible real entities (locations, people, objects, etc.)
# Add as many as you want for variety
REAL_ENTITIES = [
    "Batman",
    "Superman",
    "Julius Caesar",
    "Queen Elizabeth",
    "Leonardo da Vinci",
    "Cleopatra",
    "Mount Everest",
    "the Eiffel Tower",
    "New York City",
    "Los Angeles",
    "Tokyo",
    "Mars",
    "Alpha Centauri",
    "Atlantis",
    "Wakanda",
    "JavaScript",
    "Python",
    "C++",
    "the Stone Age",
    "the Renaissance",
    "King Arthur",
    "Galileo Galilei",
    "Michelangelo",
    "Johann Sebastian Bach",
    "Nicola Tesla",
    "Thomas Edison",
    "the Great Wall of China",
    "Shakespeare",
    "Elon Musk",
    "the Roman Empire",
    "the Middle Ages",
    "Quantum Mechanics",
    "the Turing Award",
    "the Huns",
    "the Vikings",
    "the Sahara Desert",
    "the Nile River",
    "Underworld",
    "the Holy Grail",
    "Cleopatra's Needle",
    "Atlantis",
    "the Matrix",
    "Arendelle",
    "Ancient Greece",
    "Camelot",
    "the Trojan Horse",
    "Hyperion (the tallest tree)",
    "Blackbeard",
    "Amelia Earhart",
    "Marie Curie",
    "Plato",
    "Socrates",
    "Aristotle",
    "Nintendo",
    "the Renaissance",
    "the 21st century",
    "Beyoncé",
    "Taylor Swift",
    "Spider-Man",
    "Doctor Strange",
    "Hogwarts",
    "Gotham City",
    "Zion",
    "the Dark Ages",
    # etc...
]

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##############################################################################
# Load spaCy
##############################################################################
logging.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

##############################################################################
# Ensure output directory
##############################################################################
os.makedirs(output_path, exist_ok=True)

##############################################################################
# Helper function: get named entities from text
##############################################################################
def get_entities(text):
    """
    Extract (entity_text, start_char) for each recognized entity in text.
    """
    doc = nlp(text)
    return [(ent.text, ent.start_char) for ent in doc.ents]

##############################################################################
# Helper function: naive substitution
##############################################################################
def swap_entity_in_text(text, entity, start_idx, new_entity_str):
    """
    Replaces the entity substring in `text` with @new_entity_str@, preserving
    everything else. 
    Example:
      text="Alan Turing was born in London"
      entity="London", start_idx=23, new_entity_str="@Gotham City@"
    returns -> "Alan Turing was born in @Gotham City@"
    """
    before = text[:start_idx]
    after = text[start_idx + len(entity):]
    return before + "@" + new_entity_str + "@" + after

##############################################################################
# Main function to process each split (train, valid, test)
##############################################################################
def process_data(data_type):
    """
    Reads wiki_{data_type}.json, does naive real-entity swapping, and writes
    data_{data_type}.json in the same final format as the original generate_data.py.
    """
    data_file = os.path.join(wiki_path, f"wiki_{data_type}.json")
    if not os.path.exists(data_file):
        logging.warning(f"Data file {data_file} does not exist. Skipping {data_type}.")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.info(f"Loaded {len(data)} items from {data_file}")
    result = []

    for idx_doc, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {data_type}"):
        title = item.get("title", "")
        sentences = item.get("sentences", [])
        if not sentences:
            # If no sentences, just store empty
            result.append({
                "original_text": "",
                "title": title,
                "texts": [],
                "new_entities": [],
                "original_entities": []
            })
            continue

        # Use only the first 2 sentences
        original_text = " ".join(sentences[:2])

        # NER on the text
        ents_found = get_entities(original_text)

        # Deduplicate offsets
        used_offsets = set()
        filtered_ents = []
        for (ent_str, start_char) in ents_found:
            if start_char not in used_offsets:
                used_offsets.add(start_char)
                filtered_ents.append((ent_str, start_char))

        # We'll build up "mytexts", "new_entities", "original_entities"
        mytexts = []
        new_entities = []
        original_entity = []

        # For each recognized entity
        for (ent_str, start_char) in filtered_ents:
            # skip if index=0 or entity is in the title
            if start_char == 0:
                continue
            if ent_str in title:
                continue

            # pick a random real entity from our big list
            rand_entity = random.choice(REAL_ENTITIES)
            # do the replacement
            replaced_text = swap_entity_in_text(original_text, ent_str, start_char, rand_entity)

            # If replaced_text is the same or empty, skip
            if replaced_text == original_text:
                continue

            mytexts.append(replaced_text)
            new_entities.append(rand_entity.lower())  # mimic the original approach that uses .lower()
            original_entity.append((ent_str, start_char))

        # Build the final dict
        result.append({
            "original_text": original_text,
            "title": title,
            "texts": mytexts,
            "new_entities": new_entities,
            "original_entities": original_entity
        })

    # Save
    output_file = os.path.join(output_path, f"data_{data_type}.json")
    with open(output_file, "w", encoding="utf-8") as fw:
        json.dump(result, fw, indent=4)
    logging.info(f"Saved swapped data to {output_file}")

##############################################################################
# Entry point
##############################################################################
if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        process_data(split)
