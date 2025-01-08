import os
import json
import re
import sys
import logging
from typing import List, Optional, Union
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from bio_gpt_utils import BioGPTGenerator
from retrieval import hybrid_retrieve
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# Suppress specific warnings from transformers
def suppress_transformers_logging():
    logger = logging.getLogger("transformers")
    logger.setLevel(logging.ERROR)

# Context manager to suppress specific stderr messages
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
        # Retrieve the suppressed messages
        suppressed = sys.stderr.getvalue()
        sys.stderr = original_stderr
        # Filter out unwanted messages
        for pattern in patterns:
            suppressed = re.sub(pattern, '', suppressed, flags=re.MULTILINE)
        # Print the remaining messages if any
        if suppressed.strip():
            print(suppressed, file=sys.stderr)

# Define the Tee class to duplicate stdout to terminal and file
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
    """
    Cleans the exact_answer by flattening lists, stripping whitespace,
    and replacing multiple spaces with a single space.
    """
    if exact_answer is None:
        return ""
    if isinstance(exact_answer, list):
        exact_answer = " ".join([
            item for sublist in exact_answer
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ])
    return re.sub(r"\s+", " ", exact_answer.strip())

def clean_rag_answer(rag_answer: str) -> str:
    """
    Cleans the RAG-generated answer by removing noise such as author notes,
    references, incomplete sentences, and other extraneous text.
    """
    # Remove text after special characters often indicating noise
    rag_answer = re.split(r'▃|< /|•|< FREETEXT', rag_answer)[0]

    # Remove sentences that start with common noise phrases
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

    # Remove multiple spaces and trim
    rag_answer = re.sub(r"\s+", " ", rag_answer).strip()

    return rag_answer

def yes_no_exact_match(gold: str, pred: str) -> float:
    """
    Quick hacky measure for yes/no questions: returns 1.0 if they match exactly, else 0.
    Lowercase both sides for comparison.
    """
    gold = gold.strip().lower()
    pred = pred.strip().lower()
    # We look for "yes" or "no" in the gold
    if gold in ["yes", "no"]:
        return 1.0 if gold == pred else 0.0
    # fallback if gold isn't strictly "yes"/"no"
    return 0.0

def generate_prompt(question: str, answer_type: str = "direct") -> str:
    """
    Generates a tuned prompt based on the question and expected answer type.
    """
    if answer_type == "yes_no":
        prompt = f"Question: {question}\nAnswer (Yes/No):"
    elif answer_type == "list":
        prompt = f"Question: {question}\nAnswer (List all applicable items separated by commas):"
    else:
        prompt = f"Question: {question}\nAnswer:"
    return prompt

def test_bioasq_with_biogpt(
    bioasq_json_path: str = "bioASQ/simple_ID_filtered_bioASQ.json",
    max_questions: Optional[int] = None
):
    if not os.path.exists(bioasq_json_path):
        print(f"[ERROR] Could not find: {bioasq_json_path}")
        return

    with open(bioasq_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    num_questions = len(questions)
    print(f"[INFO] Loaded {num_questions} questions from {bioasq_json_path}")

    # Initialize BioGPT
    bio_gpt = BioGPTGenerator()

    # Accumulators for overall scores
    rouge_scores_raw = []
    rouge_scores_rag = []
    bert_scores_raw = []
    bert_scores_rag = []
    yes_no_matches_raw = []
    yes_no_matches_rag = []

    # Suppress transformers logging
    suppress_transformers_logging()

    for i, q in enumerate(questions):
        if max_questions is not None and i >= max_questions:
            break

        question_id = q.get("id", "N/A")
        question_text = q.get("body", "N/A")
        exact_answer = clean_exact_answer(q.get("exact_answer", []))
        ideal_answer = " ".join(q.get("ideal_answer", [])) if q.get("ideal_answer") else ""

        print("\n" + "=" * 80)
        print(f"Question #{i + 1} | ID: {question_id}")
        print(f"Q: {question_text}")
        print(f"Exact Answer (BioASQ gold): {exact_answer}")
        print(f"Ideal Answer (BioASQ gold): {ideal_answer}")

        # Determine answer type for prompt tuning
        answer_type = "direct"
        if exact_answer.lower() in ["yes", "no"]:
            answer_type = "yes_no"
        elif re.search(r'\b(list|which|name)\b', q.get("body", "").lower()):
            answer_type = "list"

        # Generate prompt
        prompt = generate_prompt(question_text, answer_type)

        # A) BioGPT Raw
        print("\n-- BioGPT (Raw) --")
        with suppress_specific_stderr([
            r"Some weights of RobertaModel.*",
            r"You should probably TRAIN",
            r"Asking to truncate to max_length"
        ]):
            bio_gpt_raw_answer = bio_gpt.generate_raw_answer(prompt, max_length=200)
        print(bio_gpt_raw_answer)

        # Evaluate Raw w/ ROUGE + BERTScore
        if exact_answer:
            rouge_raw = calculate_rouge(exact_answer, bio_gpt_raw_answer)
            bert_raw = calculate_bertscore(exact_answer, bio_gpt_raw_answer)
            rouge_scores_raw.append(rouge_raw)
            bert_scores_raw.append(bert_raw)
        else:
            rouge_raw = {}
            bert_raw = {}
            print("\nWarning: Missing gold exact answer. Skipping evaluation for Raw.")

        # For yes/no questions, do a quick check
        if answer_type == "yes_no" and exact_answer.lower() in ["yes", "no"]:
            yes_no_matches_raw.append(yes_no_exact_match(exact_answer, bio_gpt_raw_answer))

        # B) BioGPT + RAG
        print("\n-- BioGPT (RAG) --")
        with suppress_specific_stderr([
            r"Some weights of RobertaModel.*",
            r"You should probably TRAIN",
            r"Asking to truncate to max_length"
        ]):
            relevant_docs = hybrid_retrieve(question_text, top_k=5) 
            bio_gpt_rag_answer = bio_gpt.generate_rag_answer(prompt, relevant_docs, max_new_tokens=80)
        # Clean the RAG answer
        bio_gpt_rag_answer = clean_rag_answer(bio_gpt_rag_answer)
        print(bio_gpt_rag_answer)

        # Evaluate RAG
        if exact_answer:
            rouge_rag = calculate_rouge(exact_answer, bio_gpt_rag_answer)
            bert_rag = calculate_bertscore(exact_answer, bio_gpt_rag_answer)
            rouge_scores_rag.append(rouge_rag)
            bert_scores_rag.append(bert_rag)
        else:
            rouge_rag = {}
            bert_rag = {}
            print("Warning: Missing gold exact answer. Skipping evaluation for RAG.")

        if answer_type == "yes_no" and exact_answer.lower() in ["yes", "no"]:
            yes_no_matches_rag.append(yes_no_exact_match(exact_answer, bio_gpt_rag_answer))

        # Show partial metrics
        print("\n[ROUGE Scores]")
        print(f"Raw: {rouge_raw} | RAG: {rouge_rag}")

        print("\n[BERT Scores]")
        print(f"Raw: {bert_raw} | RAG: {bert_rag}")

        print("=" * 80)

    # Summaries
    num_evaluated = len(rouge_scores_raw)
    if num_evaluated == 0:
        print("No questions evaluated.")
        return

    # Calculate average ROUGE and BERT scores
    def average_scores(scores: List[dict], metric: str) -> dict:
        avg = {}
        for key in scores[0].keys():
            avg[key] = {
                'precision': sum(s[key].precision for s in scores) / len(scores),
                'recall': sum(s[key].recall for s in scores) / len(scores),
                'fmeasure': sum(s[key].fmeasure for s in scores) / len(scores)
            }
        return avg

    average_rouge_raw = average_scores(rouge_scores_raw, "rouge1")
    average_rouge_rag = average_scores(rouge_scores_rag, "rouge1")
    average_bert_raw = {
        'precision': sum(s['precision'] for s in bert_scores_raw) / len(bert_scores_raw),
        'recall': sum(s['recall'] for s in bert_scores_raw) / len(bert_scores_raw),
        'f1': sum(s['f1'] for s in bert_scores_raw) / len(bert_scores_raw)
    } if bert_scores_raw else {}
    average_bert_rag = {
        'precision': sum(s['precision'] for s in bert_scores_rag) / len(bert_scores_rag),
        'recall': sum(s['recall'] for s in bert_scores_rag) / len(bert_scores_rag),
        'f1': sum(s['f1'] for s in bert_scores_rag) / len(bert_scores_rag)
    } if bert_scores_rag else {}

    # Optional: average ROUGE by metric
    print("\n" + "=" * 80)
    print("[AVERAGE ROUGE SCORES]")
    print(f"Raw: {average_rouge_raw} | RAG: {average_rouge_rag}")

    print("\n[AVERAGE BERTScores]")
    print(f"Raw: {average_bert_raw} | RAG: {average_bert_rag}")

    # Yes/No Accuracy
    if yes_no_matches_raw:
        yn_accuracy_raw = sum(yes_no_matches_raw) / len(yes_no_matches_raw)
        yn_accuracy_rag = sum(yes_no_matches_rag) / len(yes_no_matches_rag) if yes_no_matches_rag else 0.0
        print(f"\n[Yes/No Accuracy] Raw: {yn_accuracy_raw:.2f} | RAG: {yn_accuracy_rag:.2f}")

    print("\n[RESULTS]")
    print(f"Tested {num_evaluated} questions.")
    print("=" * 80)

def main():
    # Redirect stdout to both terminal and run_results.txt
    with open("run_results/run_results_bioASQ_ID.txt", "w", encoding="utf-8") as f_out:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f_out)
        try:
            test_bioasq_with_biogpt(
                bioasq_json_path="bioASQ/simple_ID_filtered_bioASQ.json",
                max_questions=169  # Adjust as needed
            )
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()
