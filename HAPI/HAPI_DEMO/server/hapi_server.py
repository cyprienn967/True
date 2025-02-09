# server/classifier.py
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class HallucinationClassifier(nn.Module):
    def __init__(self, input_size):
        super(HallucinationClassifier, self).__init__()
        # Use an OrderedDict to explicitly name the layers
        self.model = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_size, 256)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(256, 128)),
            ("relu2", nn.ReLU()),
            ("linear3", nn.Linear(128, 64)),
            ("relu3", nn.ReLU()),
            ("linear4", nn.Linear(64, 2))
        ]))
    
    def forward(self, x):
        return self.model(x)

class HAPIClassifier:
    def __init__(self, classifier_path, device="cpu", input_size=8192):
        self.device = device
        self.model = HallucinationClassifier(input_size)
        # Load the checkpoint (using weights_only default is False, but that warning is safe if you trust your file)
        checkpoint = torch.load(classifier_path, map_location=self.device)
        # Now the keys in checkpoint["model_state_dict"] should match our model's keys.
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, features):
        """
        Given a feature vector (list of floats), returns the hallucination probability.
        The probability for class index 1 (hallucination) is returned.
        """
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            score = probs[0, 1].item()
        return score
