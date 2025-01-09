import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
            nn.Linear(hidden_size // 2, 2)  # 2 classes
        )
    def forward(self, x):
        return self.net(x)

class HallucinationDetector:
    def __init__(self,
                 mlp_path="data/best_mlp_model.pt",
                 device="cpu",
                 model_name="microsoft/biogpt",
                 hidden_dim=1024):
        
        self.device = device

        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.biogpt = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.biogpt.eval()

        
        self.mlp = SimpleMLP(input_size=2*hidden_dim)  
        self.mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        self.mlp.to(device)
        self.mlp.eval()

        self.hidden_dim = hidden_dim

    def _extract_embeddings(self, text: str):
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.biogpt(**inputs)

        # [1, seq_len, hidden_dim]
        last_hidden = outputs.last_hidden_state
        last_token_emb = last_hidden[:, -1, :]  # shape [1, hidden_dim]

        # Last 4 layers
        hidden_states = outputs.hidden_states  # tuple of length (#layers+1)
        last_4 = hidden_states[-4:]            # each [1, seq_len, hidden_dim]
        stacked = torch.stack(last_4, dim=0)   # shape [4, 1, seq_len, hidden_dim]
        mean_4 = torch.mean(stacked, dim=0)    # [1, seq_len, hidden_dim]
        seq_mean = mean_4.mean(dim=1)          # shape [1, hidden_dim]

        # Concat
        combined = torch.cat([last_token_emb, seq_mean], dim=1)  # shape [1, 2*hidden_dim]
        return combined

    def is_hallucinated(self, text: str, threshold=0.5) -> bool:
        """Returns True if predicted label=1 (hallucinated)."""
        emb = self._extract_embeddings(text)
        with torch.no_grad():
            logits = self.mlp(emb)  # shape [1, 2]
            probs = torch.softmax(logits, dim=1)[:, 1]
            hallu_prob = probs.item()
        return (hallu_prob >= threshold)
