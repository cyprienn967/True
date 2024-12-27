import json
from datasets import load_dataset
from transformers import pipeline
import evaluate

def evaluate_bias(model_name="gpt2"):
    # Load dataset
    bold = load_dataset("AlexaAI/bold", split="train")
    prompts = bold.shuffle(seed=42).select(range(50))  # Limit to 50 prompts for efficiency

    # Load text generation pipeline
    text_gen = pipeline(
        "text-generation",
        model=model_name,
        pad_token_id=50256  # Explicitly set pad_token_id
    )

    # Generate outputs
    continuations = []
    for p in prompts:
        prompt = p.get("prompts", [])[0]  # Get the first prompt from the list
        if isinstance(prompt, str):
            continuation = text_gen(
                prompt,
                max_length=50,
                truncation=True,  # Explicit truncation
                do_sample=False
            )
            continuations.append({
                "prompt": prompt,
                "generated_text": continuation[0]["generated_text"],
                "reference": p["category"]  # Add category as the reference
            })
        else:
            print(f"Skipping invalid prompt: {p}")

    # Extract predictions and references
    generated_texts = [entry["generated_text"] for entry in continuations]
    references = [entry["reference"] for entry in continuations]

    # Evaluate bias
    regard_metric = evaluate.load("regard", "compare")
    bias_results = regard_metric.compute(data=generated_texts, references=references)

    # Save results
    filename = f"results/bias_evaluation_{model_name}.json"
    results = {
        "generated_data": continuations,
        "bias_results": bias_results
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    return filename

if __name__ == "__main__":
    output_file = evaluate_bias()
    print(f"Results saved to {output_file}")
