import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class HallucinationClassifier:
    def __init__(self, classifier_path, device="cpu", input_size=8192):
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.model.load_state_dict(torch.load(classifier_path, map_location=self.device)["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, hidden_states):
        """Returns True if hallucination is detected, False otherwise"""
        with torch.no_grad():
            input_tensor = torch.tensor(hidden_states, dtype=torch.float32).to(self.device).unsqueeze(0)
            scores = self.model(input_tensor)
            probabilities = F.softmax(scores, dim=1)
            return probabilities[0, 1].item() > 0.5  # If hallu prob > 0.5, consider it a hallucination
