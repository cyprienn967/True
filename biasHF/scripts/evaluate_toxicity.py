import json
from datasets import load_dataset
from transformers import pipeline
import evaluate

def evaluate_toxicity(model_name="gpt2"):
    # Load dataset
    toxicity_prompts = load_dataset("allenai/real-toxicity-prompts", split="train")
    toxic_sample = toxicity_prompts.shuffle(seed=42).select(range(100))
    
    # Load text generation pipeline
    text_gen = pipeline("text-generation", model=model_name)
    
    # Generate outputs
    continuations = []
    for p in toxic_sample:
        # Extract the actual text from the nested dictionary
        prompt = p.get("prompt", {}).get("text", "")
        if isinstance(prompt, str):
            continuation = text_gen(prompt, max_length=50, truncation=True, do_sample=False, pad_token_id=50256)
            continuations.append({
                "prompt": prompt,
                "generated_text": continuation[0]["generated_text"]
            })
        else:
            print(f"Skipping invalid prompt: {p}")
    
    # Evaluate toxicity
    toxicity_metric = evaluate.load("toxicity")
    generated_texts = [entry["generated_text"] for entry in continuations]
    toxicity_results = toxicity_metric.compute(predictions=generated_texts)

    # Save results
    filename = f"results/toxicity_evaluation_{model_name}.json"
    results = {
        "generated_data": continuations,
        "toxicity_results": toxicity_results
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    return filename

if __name__ == "__main__":
    output_file = evaluate_toxicity()
    print(f"Results saved to {output_file}")
