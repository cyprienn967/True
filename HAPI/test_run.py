# test_run.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        model_path = "EleutherAI/gpt-neo-1.3B"
        logging.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, return_dict_in_generate=True)
        model.to("cpu")  # Ensure the model runs on CPU

        input_text = "Artificial intelligence is transforming the world by"
        logging.info(f"Encoding input text: '{input_text}'")
        inputs = tokenizer(input_text, return_tensors="pt")

        logging.info("Generating output...")
        outputs = model.generate(
            **inputs,
            max_length=50,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Access the generated sequences
        generated_sequences = outputs.sequences
        generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
        print("Generated Text:")
        print(generated_text)

        # Access hidden states
        logging.info("Extracting hidden states...")
        with torch.no_grad():
            outputs_forward = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs_forward.hidden_states
        print(f"Number of hidden layers: {len(hidden_states)}")
        print(f"Shape of last hidden state: {hidden_states[-1].shape}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
