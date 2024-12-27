from scripts.evaluate_toxicity import evaluate_toxicity
from scripts.evaluate_bias import evaluate_bias
from scripts.evaluate_honest import evaluate_honest

def main():
    model_name = "gpt2"  # You can change the model name here

    print("Evaluating Toxicity...")
    toxicity_file = evaluate_toxicity(model_name)
    print(f"Toxicity results saved to {toxicity_file}")

    print("Evaluating Bias (Regard)...")
    bias_file = evaluate_bias(model_name)
    print(f"Bias results saved to {bias_file}")

    #print("Evaluating Hurtfulness (HONEST)...")
    #honest_file = evaluate_honest(model_name)
    #print(f"HONEST results saved to {honest_file}")

if __name__ == "__main__":
    main()
