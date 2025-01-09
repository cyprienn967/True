import os
import json
import re
import sys
import logging
from typing import List, Optional, Union
import requests

from bert_score import score as bert_score
from rouge_score import rouge_scorer

from contextlib import contextmanager
import io
import numpy as np


def suppress_transformers_logging():
    logger = logging.getLogger("transformers")
    logger.setLevel(logging.ERROR)

@contextmanager
def suppress_specific_stderr(patterns: List[str]):
    """
    Suppresses stderr messages that match any of the given regex patterns.
    """
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        suppressed = sys.stderr.getvalue()
        sys.stderr = original_stderr
        for pattern in patterns:
            suppressed = re.sub(pattern, '', suppressed, flags=re.MULTILINE)
        if suppressed.strip():
            print(suppressed, file=sys.stderr)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def calculate_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def calculate_bertscore(reference: str, hypothesis: str) -> dict:
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

def clean_exact_answer(exact_answer: Union[str, List[str], None]) -> str:

    if exact_answer is None:
        return ""
    if isinstance(exact_answer, list):
        joined = []
        for item in exact_answer:
            if isinstance(item, list):
                joined += item
            else:
                joined.append(item)
        exact_answer = " ".join(joined)
    return re.sub(r"\s+", " ", exact_answer.strip())

def clean_rag_answer(rag_answer: str) -> str:
    """
    Clean up the generated answer removing noise, references, etc.
    """
    rag_answer = re.split(r'▃|< /|•|< FREETEXT', rag_answer)[0]
    noise_phrases = [
        r"Some weights of RobertaModel",
        r"You should probably TRAIN",
        r"Author response",
        r"Reviewer",
        r"This work was supported",
        r"References",
        r"We have added",
        r"We thank",
        r"The authors declare",
        r"Please click here",
        r"Essential revisions",
        r"Asking to truncate to max_length"
    ]
    for phrase in noise_phrases:
        rag_answer = re.sub(rf'.*{phrase}.*', '', rag_answer, flags=re.IGNORECASE)
    rag_answer = re.sub(r"\s+", " ", rag_answer).strip()
    return rag_answer

def yes_no_exact_match(gold: str, pred: str) -> float:
    gold = gold.strip().lower()
    pred = pred.strip().lower()
    if gold in ["yes", "no"]:
        return 1.0 if gold == pred else 0.0
    return 0.0

def test_bioasq_with_biogpt(
    bioasq_json_path: str = "bioASQ/simple_ID_filtered_bioASQ.json",
    max_questions: Optional[int] = None
):

    print("** Placeholder for your existing test_bioasq_with_biogpt code **")
    pass

def benchmark_hapi_vs_raw(
    bioasq_json_path: str,
    hapi_endpoint: str = "http://localhost:8000/hapi/hapi_generate",
    max_questions: Optional[int] = 5
):

    if not os.path.exists(bioasq_json_path):
        print(f"[ERROR] Could not find: {bioasq_json_path}")
        return

    with open(bioasq_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    print(f"[INFO] Loaded {len(questions)} questions from {bioasq_json_path}")

    # accumulators
    rouge_scores_raw = []
    bert_scores_raw = []
    rouge_scores_hapi = []
    bert_scores_hapi = []

    for i, q in enumerate(questions):
        if max_questions is not None and i >= max_questions:
            break

        question_text = q.get("body", "")
        ideal_ans_list = q.get("ideal_answer", [])
        ideal_str = " ".join(ideal_ans_list).strip()

        if not ideal_str:
            print(f"[WARNING] No ideal answer for question #{i}, skipping.")
            continue

        
        payload_raw = {
            "question": question_text,
            "conversation_id": f"hapi-bench-{i}",
            "with_hallucination_filter": False
        }
        resp_raw = requests.post(hapi_endpoint, json=payload_raw)
        raw_answer = ""
        if resp_raw.status_code == 200:
            raw_data = resp_raw.json()
            raw_answer = raw_data["final_answer"]
        else:
            print(f"[ERROR] /hapi_generate raw request failed: {resp_raw.text}")

        
        payload_hapi = {
            "question": question_text,
            "conversation_id": f"hapi-bench-{i}",
            "with_hallucination_filter": True
        }
        resp_hapi = requests.post(hapi_endpoint, json=payload_hapi)
        hapi_answer = ""
        if resp_hapi.status_code == 200:
            hapi_data = resp_hapi.json()
            hapi_answer = hapi_data["final_answer"]
        else:
            print(f"[ERROR] /hapi_generate hapi request failed: {resp_hapi.text}")

        
        raw_answer = clean_rag_answer(raw_answer)
        hapi_answer = clean_rag_answer(hapi_answer)

        
        rouge_raw = calculate_rouge(ideal_str, raw_answer)
        bert_raw = calculate_bertscore(ideal_str, raw_answer)

        rouge_hapi = calculate_rouge(ideal_str, hapi_answer)
        bert_hapi = calculate_bertscore(ideal_str, hapi_answer)

        rouge_scores_raw.append(rouge_raw)
        bert_scores_raw.append(bert_raw)
        rouge_scores_hapi.append(rouge_hapi)
        bert_scores_hapi.append(bert_hapi)

        print(f"\nQ#{i}: {question_text[:50]}...")
        print(f"Raw => {raw_answer}")
        print(f"HAPI => {hapi_answer}")
        print("[ROUGE RAW]:", rouge_raw)
        print("[ROUGE HAPI]:", rouge_hapi)
        print("[BERT RAW]:", bert_raw)
        print("[BERT HAPI]:", bert_hapi)

    
    def average_rouge(scores_list):
        if not scores_list:
            return {}
        metrics = ["rouge1", "rouge2", "rougeL"]
        avg = {}
        for metric in metrics:
            precision = np.mean([s[metric].precision for s in scores_list])
            recall = np.mean([s[metric].recall for s in scores_list])
            fmeasure = np.mean([s[metric].fmeasure for s in scores_list])
            avg[metric] = {
                "precision": precision,
                "recall": recall,
                "fmeasure": fmeasure
            }
        return avg

    def average_bert(scores_list):
        if not scores_list:
            return {}
        p = np.mean([s["precision"] for s in scores_list])
        r = np.mean([s["recall"] for s in scores_list])
        f = np.mean([s["f1"] for s in scores_list])
        return {"precision": p, "recall": r, "f1": f}

    avg_rouge_raw = average_rouge(rouge_scores_raw)
    avg_rouge_hapi = average_rouge(rouge_scores_hapi)
    avg_bert_raw = average_bert(bert_scores_raw)
    avg_bert_hapi = average_bert(bert_scores_hapi)

    print("\n======== [BENCHMARK RESULTS: RAW vs. HAPI] ========")
    print(f"[Avg ROUGE - RAW ]: {avg_rouge_raw}")
    print(f"[Avg ROUGE - HAPI]: {avg_rouge_hapi}")
    print(f"[Avg BERT  - RAW ]: {avg_bert_raw}")
    print(f"[Avg BERT  - HAPI]: {avg_bert_hapi}")
    print("===================================================")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hapi", action="store_true",
                        help="If set, run benchmark_hapi_vs_raw. Otherwise run test_bioasq_with_biogpt.")
    parser.add_argument("--bioasq_json_path", default="bioASQ/simple_ID_filtered_bioASQ.json")
    parser.add_argument("--max_questions", type=int, default=5)
    args = parser.parse_args()

    if args.use_hapi:
        
        benchmark_hapi_vs_raw(args.bioasq_json_path, max_questions=args.max_questions)
    else:
        print("[INFO] Running original pipeline test_bioasq_with_biogpt ...")
        test_bioasq_with_biogpt(args.bioasq_json_path, max_questions=args.max_questions)

if __name__ == "__main__":
    main()
