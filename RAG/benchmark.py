import os
import json
import re
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from bio_gpt_utils import BioGPTGenerator
from retrieval import hybrid_retrieve

def calculate_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)

def calculate_bertscore(reference: str, hypothesis: str) -> dict:
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

def clean_exact_answer(exact_answer):
    if exact_answer is None:
        return ""
    if isinstance(exact_answer, list):
        exact_answer = " ".join([
            item for sublist in exact_answer
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ])
    return re.sub(r"\s+", " ", exact_answer.strip())


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

def test_bioasq_with_biogpt(
    bioasq_json_path: str = "bioASQ/simple_ID_filtered_bioASQ.json",
    max_questions: int = None
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

        # A) BioGPT Raw
        bio_gpt_raw_answer = bio_gpt.generate_raw_answer(question_text, max_length=200)
        print("\n-- BioGPT (Raw) --")
        print(bio_gpt_raw_answer)

        # Evaluate Raw w/ ROUGE + BERTScore
        rouge_raw = calculate_rouge(exact_answer, bio_gpt_raw_answer)
        bert_raw = calculate_bertscore(exact_answer, bio_gpt_raw_answer)
        rouge_scores_raw.append(rouge_raw)
        bert_scores_raw.append(bert_raw)

        # For yes/no questions, do a quick check
        if exact_answer.lower() in ["yes", "no"]:
            yes_no_matches_raw.append(yes_no_exact_match(exact_answer, bio_gpt_raw_answer))

        # B) BioGPT + RAG
        relevant_docs = hybrid_retrieve(question_text, top_k=5) 
        bio_gpt_rag_answer = bio_gpt.generate_rag_answer(question_text, relevant_docs, max_new_tokens=80)
        print("\n-- BioGPT (RAG) --")
        print(bio_gpt_rag_answer)

        # Evaluate RAG
        rouge_rag = calculate_rouge(exact_answer, bio_gpt_rag_answer)
        bert_rag = calculate_bertscore(exact_answer, bio_gpt_rag_answer)
        rouge_scores_rag.append(rouge_rag)
        bert_scores_rag.append(bert_rag)

        if exact_answer.lower() in ["yes", "no"]:
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

    # Optional: average the BERT/ROUGE
    # (rouge_raw is a dict of 3 keys, so we can do a simplistic average if we want)
    # We'll just show how many yes/no matches we got:
    if len(yes_no_matches_raw) > 0:
        yn_accuracy_raw = sum(yes_no_matches_raw) / len(yes_no_matches_raw)
        yn_accuracy_rag = sum(yes_no_matches_rag) / len(yes_no_matches_rag)
        print(f"\n[Yes/No Accuracy] Raw: {yn_accuracy_raw:.2f} | RAG: {yn_accuracy_rag:.2f}")

    print("\n[RESULTS]")
    print(f"Tested {num_evaluated} questions.")
    # We won't re-print all the ROUGE dicts for brevity.
    print("Done.")


def main():
    test_bioasq_with_biogpt(
        bioasq_json_path="bioASQ/simple_ID_filtered_bioASQ.json",
        max_questions=169  # or any subset
    )

if __name__ == "__main__":
    main()
