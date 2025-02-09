# client/generate_without_hapi.py
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from huggingface_hub import login
login(token="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")


def generate_text(prompt, model, tokenizer, max_new_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_name = "meta-llama/Llama-2-7b-hf"  # Adjust if needed (or provide a local path)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Read prompts
    with open("prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    output_file = "generation_without_hapi.txt"
    with open(output_file, "w", encoding="utf-8") as out_f:
        for prompt in prompts:
            generated = generate_text(prompt, model, tokenizer)
            out_f.write(f"Prompt: {prompt}\nGenerated: {generated}\n{'-'*40}\n")
            print(f"Generated text for prompt: {prompt}")
    
    print(f"Generation without HAPI saved to {output_file}")

if __name__ == "__main__":
    main()
