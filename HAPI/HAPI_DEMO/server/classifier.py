# server/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HallucinationClassifier(nn.Module):
    def __init__(self, input_size):
        super(HallucinationClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.model(x)

class HAPIClassifier:
    def __init__(self, classifier_path, device="cpu", input_size=8192):
        self.device = device
        self.model = HallucinationClassifier(input_size)
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, features):
        """
        Given a list of features, returns the hallucination probability.
        We assume the output class index 1 corresponds to hallucination.
        """
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            score = probs[0, 1].item()
        return score
