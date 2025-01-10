# File: HAPI/scripts2/benchmark.py

import os
import json
import requests
import numpy as np
import sys
import re
from typing import Dict
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class Tee:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def compute_rouge(reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def compute_bertscore_metric(reference: str, hypothesis: str):
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}

def strip_prompt(generated_text: str, question_str: str) -> str:
    """
    Removes the 'QUESTION? Answer the question factually...' portion and any subsequent instructions
    from the generated text, leaving only the actual answer.
    """
    possible_prompt_subs = [
        f"{question_str}\nAnswer the question factually and concisely without any references:",
        f"{question_str} Answer the question factually and concisely without any references:",
        f"{question_str}\nAnswer:",
        f"{question_str} Answer:",
    ]
    out = generated_text
    for subs in possible_prompt_subs:
        idx = out.lower().find(subs.lower())
        if idx != -1:
            out = out[idx + len(subs):].strip()
            break
    else:
        # If none of the prompt variations are found, return the original text
        return out

    # Remove any instructions or text before the actual answer.
    # Assuming that actual answer starts after the last colon.
    # This handles cases where instructions are followed by a colon.
    # Example: "...responses.: The actual answer starts here."
    last_colon = out.rfind(':')
    if last_colon != -1:
        out = out[last_colon + 1:].strip()

    # Additionally, remove any lingering unwanted phrases
    out = remove_unwanted_phrases(out)

    return out

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

def avg_rouge(scores):
    if not scores:
        return {}
    metrics = ["rouge1","rouge2","rougeL"]
    result = {}
    for m in metrics:
        p = np.mean([sc[m].precision for sc in scores])
        r = np.mean([sc[m].recall for sc in scores])
        f = np.mean([sc[m].fmeasure for sc in scores])
        result[m] = {"precision": round(p, 4), "recall": round(r,4), "fmeasure": round(f,4)}
    return result

def avg_bert(scores):
    if not scores:
        return {}
    pr = np.mean([s["precision"] for s in scores])
    re = np.mean([s["recall"]    for s in scores])
    f1 = np.mean([s["f1"]        for s in scores])
    return {"precision": round(pr,4), "recall": round(re,4), "f1": round(f1,4)}

def main():
    data_file = "../data/questions.json"
    if not os.path.exists(data_file):
        print(f"[ERROR] {data_file} not found.")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] Loaded {len(dataset)} questions from {data_file}")

    # Define accumulators for ROUGE/BERT
    rouge_raw_list = []
    rouge_hapi_list= []
    bert_raw_list  = []
    bert_hapi_list = []

    for i, item in enumerate(dataset, 1):
        question = item.get("question", "").strip()
        ideal    = item.get("ideal_answer", "").strip()
        if not (question and ideal):
            continue

        print(f"\n=== Q#{i} ===")
        print("Question:", question)
        print("Ideal:", ideal)

        # 1) RAW - Direct BioGPT Generation
        try:
            resp_raw = requests.post("http://localhost:8000/generate", json={
                "question": f"{question} Answer the question factually and concisely without any references:"  
            }, timeout=300)  # Adjust timeout as necessary
            if resp_raw.status_code == 200:
                raw_ans = resp_raw.json().get("final_answer", "")
            else:
                print("[ERROR] HAPI request failed (raw):", resp_raw.text)
                raw_ans = ""
        except requests.exceptions.RequestException as e:
            print("[ERROR] HAPI request failed (raw):", str(e))
            raw_ans = ""

        # 2) HAPI - Generation with Hallucination Detection
        try:
            resp_hapi = requests.post("http://localhost:8000/generate", json={
                "question": f"{question} Answer the question factually and concisely without any references:",
                "with_hallucination_filter": True  # Assuming the API can handle this flag
            }, timeout=300)
            if resp_hapi.status_code == 200:
                hapi_ans = resp_hapi.json().get("final_answer", "")
            else:
                print("[ERROR] HAPI request failed (hapi):", resp_hapi.text)
                hapi_ans = ""
        except requests.exceptions.RequestException as e:
            print("[ERROR] HAPI request failed (hapi):", str(e))
            hapi_ans = ""

        # Print the raw outputs
        print("[RAW Answer]:", raw_ans)
        print("[HAPI Answer]:", hapi_ans)

        # Strip the prompts
        raw_ans_clean  = strip_prompt(raw_ans,  question)
        hapi_ans_clean = strip_prompt(hapi_ans, question)

        # Apply comprehensive cleaning
        raw_ans_clean  = comprehensive_cleaning(raw_ans_clean)
        hapi_ans_clean = comprehensive_cleaning(hapi_ans_clean)

        # Print cleaned answers
        print("[CLEAN RAW Answer]:", raw_ans_clean)
        print("[CLEAN HAPI Answer]:", hapi_ans_clean)

        # Compute ROUGE and BERT scores
        if raw_ans_clean:
            r_raw = compute_rouge(ideal, raw_ans_clean)
            b_raw = compute_bertscore_metric(ideal, raw_ans_clean)
            rouge_raw_list.append(r_raw)
            bert_raw_list.append(b_raw)
            print("[RAW ROUGE]:", {k: round(v,4) for k, v in r_raw.items()})
            print("[RAW BERT ]:", {k: round(v,4) for k, v in b_raw.items()})

        if hapi_ans_clean:
            r_hapi = compute_rouge(ideal, hapi_ans_clean)
            b_hapi = compute_bertscore_metric(ideal, hapi_ans_clean)
            rouge_hapi_list.append(r_hapi)
            bert_hapi_list.append(b_hapi)
            print("[HAPI ROUGE]:", {k: round(v,4) for k, v in r_hapi.items()})
            print("[HAPI BERT ]:", {k: round(v,4) for k, v in b_hapi.items()})

    # After all questions, compute average metrics
    final_rouge_raw  = avg_rouge(rouge_raw_list)
    final_rouge_hapi = avg_rouge(rouge_hapi_list)
    final_bert_raw   = avg_bert(bert_raw_list)
    final_bert_hapi  = avg_bert(bert_hapi_list)

    print("\n===== AGGREGATE METRICS =====")
    print("[ROUGE RAW ]:", final_rouge_raw)
    print("[ROUGE HAPI]:", final_rouge_hapi)
    print("[BERT RAW ]:", final_bert_raw)
    print("[BERT HAPI]:", final_bert_hapi)

if __name__ == "__main__":
    # Tee output to console + output.txt
    tee = Tee("output.txt", mode="w")
    orig_stdout = sys.stdout
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = orig_stdout
        tee.file.close()
