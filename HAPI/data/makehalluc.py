import json
import random

import spacy
import scispacy

###############################################################################
# A dictionary of scispaCy models you want to use for different question types
# or domains. The keys can be any string you use to identify the question type,
# and the values are the actual model names. 
#
# For demonstration:
#  - "disease" -> "en_ner_bc5cdr_md" (recognizes DISEASE, CHEMICAL)
#  - "general_bio" -> "en_ner_bionlp13cg_md" (recognizes GENE_OR_GENE_PRODUCT, etc.)
###############################################################################
MODEL_MAP = {
    "disease": "en_ner_bc5cdr_md",
    "general_bio": "en_ner_bionlp13cg_md"
}

###############################################################################
# A small dictionary or approach for swapping recognized entities of a certain
# type with some 'fake' placeholders. Expand this as needed.
###############################################################################
FAKE_SWAPS = {
    # For bc5cdr model (DISEASE, CHEMICAL)
    "DISEASE": ["flu", "Crohn's disease", "Parkinson's", "diabetes"],
    "CHEMICAL": ["caffeine", "aspirin", "dopamine", "ethanol"],

    # For bionlp13cg model (e.g. GENE_OR_GENE_PRODUCT, CANCER, etc.)
    "GENE_OR_GENE_PRODUCT": ["MYC", "BRCA1", "TP53", "EGFR"],
    "CANCER": ["melanoma", "leukemia", "glioblastoma"],
    "SIMPLE_CHEMICAL": ["ATP", "glucose", "lactate"]
    # ... add more as needed ...
}

###############################################################################
# Cache to store loaded models so we don't load them multiple times.
###############################################################################
model_cache = {}

def get_model_for_question(question_text: str) -> str:
    """
    Naive logic to decide which scispaCy model to use based on the question text.
    In a real product, you might parse an explicit 'question_type' or do a more
    robust classification. This is just an example:
      - If 'disease' in question, choose 'disease' model
      - Otherwise, choose 'general_bio' model
    """
    # Lowercase the question for simple matching
    q_lower = question_text.lower()

    if "disease" in q_lower or "cancer" in q_lower or "tumor" in q_lower:
        return MODEL_MAP["disease"]
    else:
        return MODEL_MAP["general_bio"]

def load_scispacy_model(model_name: str):
    """
    Load a scispaCy model from cache if available, otherwise load anew.
    """
    if model_name not in model_cache:
        print(f"Loading model: {model_name} ...")
        model_cache[model_name] = spacy.load(model_name)
    return model_cache[model_name]

def hallucinate_answer(question_text: str, ideal_answer: str) -> str:
    """
    1) Choose which scispaCy model to use based on question_text.
    2) Detect named entities in ideal_answer using that model.
    3) Swap recognized entities with random placeholders from FAKE_SWAPS.
    4) Return the hallucinated text.
    """

    model_name = get_model_for_question(question_text)
    nlp = load_scispacy_model(model_name)

    doc = nlp(ideal_answer)
    replacements = []
    
    for ent in doc.ents:
        label = ent.label_
        if label in FAKE_SWAPS:
            # pick a random 'fake' replacement
            new_ent = random.choice(FAKE_SWAPS[label])
            replacements.append((ent.start_char, ent.end_char, new_ent))

    # Sort by start index descending to do replacements safely
    replacements.sort(key=lambda x: x[0], reverse=True)

    hallu_text = ideal_answer
    for start, end, new_ent in replacements:
        hallu_text = hallu_text[:start] + new_ent + hallu_text[end:]
    
    return hallu_text

def create_hallucinated_dataset(input_json: str, output_json: str) -> None:
    """
    Reads a JSON of QAs. Each item: { q_id, question, ideal_answer, label=0 }
    For each, produces a hallucinated record (label=1) using scispaCy,
    then writes them all out.
    """

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    combined_data = []

    for record in data:
        # Keep the original record
        combined_data.append(record)

        # Generate the hallucinated version
        question_text = record.get("question", "")
        ideal_answer = record.get("ideal_answer", "")
        
        new_text = hallucinate_answer(question_text, ideal_answer)
        
        hallu_rec = {
            "q_id": record.get("q_id", ""),
            "question": question_text,
            "ideal_answer": new_text,
            "label": 1
        }
        combined_data.append(hallu_rec)

    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(combined_data, out, indent=2, ensure_ascii=False)
    
    print(f"Done! Combined dataset (orig + hallucinated) saved in: {output_json}")


if __name__ == "__main__":
    # Example usage
    create_hallucinated_dataset("questions.json", "questions_with_hallucinated.json")
