# generate_without_hapi.py
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

def main():
    model_name = "llamachat7b"  # Adjust if you have a local path or different model name
    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model.to("cuda:0")  # Google Colab typically uses cuda:0

    prompt = "The story begins with a young adventurer who"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    # Generate text (e.g., 50 new tokens)
    output = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("Generated text without HAPI:")
    print(generated_text)

if __name__ == "__main__":
    main()
