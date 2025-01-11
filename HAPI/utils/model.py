# utils/model.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model(model_type, model_family, max_new_tokens=50):
    """
    Load and configure the specified language model.

    Args:
        model_type (str): The type of the model (e.g., "1.3B").
        model_family (str): The family of the model (e.g., "gpt-neo").
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        tuple: (model, tokenizer, generation_config, at_id)
    """
    if model_family == "gpt-neo":
        model_path = "EleutherAI/gpt-neo-1.3B"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"  # Automatically maps layers to available devices (GPU if available)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        generation_config = {
            "top_p": 1.0,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,  # Ensure outputs are returned as a dict
            "output_hidden_states": True,
            "output_scores": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id
        }
        at_id = None  # GPT-Neo doesn't require at_id
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    return model, tokenizer, generation_config, at_id
