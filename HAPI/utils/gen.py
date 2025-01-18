# utils/gen.py

import torch
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define any special tokens if needed, or leave them empty
B_INST, E_INST = "", ""
B_SYS, E_SYS = "", ""

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def chat_change(dialog, tokenizer):
    """
    Convert dialog into prompt tokens.

    Args:
        dialog (list): List of message dictionaries with 'role' and 'content'.
        tokenizer (AutoTokenizer): The tokenizer instance.

    Returns:
        list: List of tokenized prompts.
    """
    prompt_tokens = []
    unsafe_requests = []
    
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    # GPT-Neo only supports 'user' role
    assert all([msg["role"] == "user" for msg in dialog]), (
        "model only supports 'user' role for GPT-Neo"
    )
    
    dialog_tokens = []
    for msg in dialog:
        dialog_tokens += tokenizer.encode(f"{msg['content'].strip()} ", add_special_tokens=False)
    
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens

def chat_change_with_answer(dialog, answer_, tokenizer):
    """
    Convert dialog and answer into prompt tokens.

    Args:
        dialog (list): List of message dictionaries with 'role' and 'content'.
        answer_ (str): The assistant's answer.
        tokenizer (AutoTokenizer): The tokenizer instance.

    Returns:
        list: List of tokenized prompts with the answer.
    """
    prompt_tokens = []
    unsafe_requests = []
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    # GPT-Neo only supports 'user' role
    assert all([msg["role"] == "user" for msg in dialog]), (
        "model only supports 'user' role for GPT-Neo"
    )
    
    dialog_tokens = []
    for msg in dialog:
        dialog_tokens += tokenizer.encode(f"{msg['content'].strip()} ", add_special_tokens=False)
    dialog_tokens += tokenizer.encode(f"{answer_.strip()} ", add_special_tokens=False)
    
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens

def generate_output(model_family, model, tokenizer, config, text, answer=None):
    """
    Generate output from the model based on the prompt.

    Args:
        model_family (str): The family of the model.
        model (AutoModelForCausalLM): The loaded model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        config (dict): Generation configuration.
        text (str): The input text prompt.
        answer (str, optional): The assistant's answer. Defaults to None.

    Returns:
        torch.Tensor: The generated output.
    """
    if "chat" not in model_family:
        assert answer is None
        input_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()
    else:
        if answer is None:
            input_id = chat_change([{"role": "user", "content": text.strip()}], tokenizer)
        else:
            input_id = chat_change_with_answer([{"role": "user", "content": text.strip()}], answer.strip(), tokenizer)
    output = model.generate(torch.tensor(input_id).to(model.device), **config)
    return output

def get_pe(logit, id_, start_at):
    """
    Calculate probabilities and entropy.

    Args:
        logit (torch.Tensor): The logit tensor from the model.
        id_ (list): List of token IDs.
        start_at (int): The starting position to calculate from.

    Returns:
        tuple: (probabilities list, entropy list)
    """
    probabilities = F.softmax(logit, dim=-1)
    log_probabilities = torch.log(probabilities)
    entropy = -probabilities * log_probabilities
    entropy_sum = torch.sum(entropy, dim=-1)

    pl = []
    el = []
    for i, idx in enumerate(id_[1:]):
        if i < start_at - 1:
            continue
        pl.append(probabilities[0][i][idx].item())
        el.append(entropy_sum[0][i].item())
    return pl, el
