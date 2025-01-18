import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def load_qa_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def extract_last_token_embedding(model, tokenizer, text, device="cpu"):

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden = outputs.last_hidden_state  
    last_token_emb = last_hidden[:, -1, :].squeeze(0) 
    return last_token_emb.cpu().numpy()

def extract_mean_of_last_4_layers(model, tokenizer, text, device="cpu"):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    # hidden_states is a tuple of length [num_layers + 1], each is [batch_size, seq_len, hidden_dim]
    # We want the last 4 layers => outputs.hidden_states[-4:]
    # e.g. if the model has 12 layers, we get the 9th, 10th, 11th, 12th layers as the last 4.
    hidden_states = outputs.hidden_states  
    last_4 = hidden_states[-4:]  

    # stack them => shape [4, 1, seq_len, hidden_dim]
    # Then mean across the first dimension (which is the layer dimension)
    # => shape [1, seq_len, hidden_dim]
    stacked = torch.stack(last_4, dim=0)
    mean_4 = torch.mean(stacked, dim=0)  # [1, seq_len, hidden_dim]

    # Now let's do a mean across seq_len for a single vector. 
    # Or you can pick the last token or the CLS token if that suits your approach better.
    # We'll do an entire sequence average for demonstration:
    # => shape [1, hidden_dim]
    seq_mean = torch.mean(mean_4, dim=1).squeeze(0)  # shape: [hidden_dim]

    return seq_mean.cpu().numpy()

def main():
    # 1. Load your data
    input_json = "../data/testq.json"  
    df = load_qa_data(input_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/biogpt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    
    last_token_embs = []
    mean_4_embs = []

    for idx, row in df.iterrows():
        text = row["ideal_answer"]
        
        # a) last token from last hidden state
        emb_last_token = extract_last_token_embedding(model, tokenizer, text, device=device)
        
        # b) average of last 4 layers
        emb_last_4 = extract_mean_of_last_4_layers(model, tokenizer, text, device=device)
        
        last_token_embs.append(emb_last_token)
        mean_4_embs.append(emb_last_4)

    # 4. Store these embeddings in the DataFrame
    df["emb_last_token"] = last_token_embs
    df["emb_mean_last4"] = mean_4_embs

    
    output_jsonl = "../data/questions_with_embeddings_test.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            rec = {
                "q_id": row["q_id"],
                "question": row["question"],
                "ideal_answer": row["ideal_answer"],
                "label": row["label"],
                "emb_last_token": row["emb_last_token"].tolist(), 
                "emb_mean_last4": row["emb_mean_last4"].tolist()
            }
            fout.write(json.dumps(rec) + "\n")
    
    print(f"Done! Wrote dataset with embeddings to {output_jsonl}")

if __name__ == "__main__":
    main()
