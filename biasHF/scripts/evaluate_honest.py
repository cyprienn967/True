import json
from datasets import load_dataset
from transformers import pipeline
import evaluate
from transformers import logging

# Suppress specific warnings
logging.set_verbosity_error()

def evaluate_honest(model_name="gpt2"):
    # Load dataset
    #honest_dataset = load_dataset("HonestQA/honest", split="validation")
    honest_dataset = load_dataset("MilaNLProc/honest", 'en_queer_nonqueer', split='honest')
    sample = honest_dataset.shuffle(seed=42).select(range(50))  # Limit to 50 samples for efficiency

    # Load text generation pipeline
    text_gen = pipeline(
        "text-generation",
        model=model_name,
        pad_token_id=50256,  # Explicitly set pad_token_id
    )

    # Generate outputs
    continuations = []
    for p in sample:
        question = p.get("question", "")
        if isinstance(question, str):
            continuation = text_gen(
                question,
                max_length=50,
                truncation=True,  # Explicit truncation
                do_sample=False
            )
            continuations.append({
                "question": question,
                "generated_text": continuation[0]["generated_text"],
                "answer": p["answer"]  # Add the correct answer for evaluation
            })
        else:
            print(f"Skipping invalid question: {p}")

    # Evaluate hurtfulness (HONEST)
    honest_metric = evaluate.load("honest")
    generated_texts = [entry["generated_text"] for entry in continuations]
    references = [entry["answer"] for entry in continuations]
    honest_results = honest_metric.compute(data=generated_texts, references=references)

    # Save results
    filename = f"results/honest_evaluation_{model_name}.json"
    results = {
        "generated_data": continuations,
        "honest_results": honest_results
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    return filename

if __name__ == "__main__":
    output_file = evaluate_honest()
    print(f"Results saved to {output_file}")
