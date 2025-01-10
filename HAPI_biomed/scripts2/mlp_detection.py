# File: HAPI/scripts2/mlp_detection.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import warnings

# Utility to suppress specific warnings (e.g., from RoBERTa)
class suppress_roberta_warning:
    def __enter__(self):
        warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.resetwarnings()

class SimpleMLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256, dropout=0.2):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.net(x)

class HallucinationDetector:
    def __init__(self,
                 mlp_path="../data/best_mlp_model.pt",
                 device="cpu",
                 model_name="microsoft/BioGPT",
                 hidden_dim=1024):
        print(f"[HallucinationDetector] Loading MLP from: {mlp_path}")
        self.device = device

        # Load BioGPT for embedding extraction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with suppress_roberta_warning():
            self.biogpt = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.biogpt.eval()

        # Load MLP
        self.mlp = SimpleMLP(input_size=2 * hidden_dim)
        try:
            mlp_state_dict = torch.load(mlp_path, map_location=device)
            self.mlp.load_state_dict(mlp_state_dict)
            print("[INFO] MLP model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load MLP model: {e}")
            exit(1)
        
        self.mlp.to(device)
        self.mlp.eval()

        self.hidden_dim = hidden_dim

    def _extract_embeddings(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}

        with suppress_roberta_warning():
            with torch.no_grad():
                outputs = self.biogpt(**inputs)

        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
        last_token_emb = last_hidden[:, -1, :]   # [1, hidden_dim]

        # last 4 layers
        hs = outputs.hidden_states
        last_4 = hs[-4:]                      # each [1, seq_len, hidden_dim]
        stacked = torch.stack(last_4, dim=0)  # [4, 1, seq_len, hidden_dim]
        mean_last4 = torch.mean(stacked, dim=0)      # [1, seq_len, hidden_dim]
        mean_last4_emb = torch.mean(mean_last4, dim=1)  # [1, hidden_dim]

        combined = torch.cat([last_token_emb, mean_last4_emb], dim=1)  # [1, 2*hidden_dim]
        return combined

    def is_hallucinated(self, text: str, threshold=0.5) -> bool:
        emb = self._extract_embeddings(text)
        with torch.no_grad():
            logits = self.mlp(emb)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (hallucination)
            prob = probs.item()
        
        return prob > threshold

# Example usage
if __name__ == "__main__":
    detector = HallucinationDetector(
        mlp_path="../data/best_mlp_model.pt",
        device="cpu",
        model_name="microsoft/BioGPT",
        hidden_dim=1024  # Adjust based on BioGPT's hidden size
    )
    
    test_text = "This is a test answer to check for hallucination."
    is_hallu = detector.is_hallucinated(test_text)
    print(f"Is hallucinated? {is_hallu}")
