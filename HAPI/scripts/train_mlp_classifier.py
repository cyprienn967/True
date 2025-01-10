import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tqdm

def load_data(jsonl_path, use_combined_vector=True):

    X = []
    y = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label = record["label"]
            last_token_vec = record["emb_last_token"]
            mean_last4_vec = record["emb_mean_last4"]
            
            if use_combined_vector:
                combined = last_token_vec + mean_last4_vec
                X.append(combined)
            else:
                X.append(last_token_vec)
            
            y.append(label)
    
    X = np.array(X, dtype=np.float32) 
    y = np.array(y, dtype=np.int64)
    return X, y

class SimpleMLP(nn.Module):

    def __init__(self, input_size, hidden_size=256, dropout=0.2):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 2)
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(X, y, epochs=5, lr=1e-3, batch_size=32, device="cpu"):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_size = X_train.shape[1]
    model = SimpleMLP(input_size=input_size).to(device)
    
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data   = torch.utils.data.TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_auc = -1
    best_model_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs = model(Xb)
                probs = torch.softmax(outputs, dim=1)[:, 1]  
                preds_all.extend(probs.cpu().tolist())
                labels_all.extend(yb.cpu().tolist())
        
        
        preds_all = np.array(preds_all)
        labels_all = np.array(labels_all)
        auc_score = roc_auc_score(labels_all, preds_all)
        pred_classes = (preds_all >= 0.5).astype(int)
        acc_score = accuracy_score(labels_all, pred_classes)
        
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, "
              f"Val AUC: {auc_score:.4f}, Val Acc: {acc_score:.4f}")
        

        if auc_score > best_val_auc:
            best_val_auc = auc_score
            best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    return model, best_val_auc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "../data/questions_with_embeddings_123.jsonl" 
    X, y = load_data(data_path, use_combined_vector=True)
    print(f"Loaded dataset. X shape={X.shape}, y shape={y.shape}, pos={np.sum(y)} neg={len(y)-np.sum(y)}")
    model, best_val_auc = train_and_evaluate(X, y, epochs=10, lr=1e-3, batch_size=32, device=device)
    print(f"Done training! Best validation AUC = {best_val_auc:.4f}")
    torch.save(model.state_dict(), "../data/best_mlp_model.pt")
    print("Model saved at ../data/best_mlp_model.pt")

if __name__ == "__main__":
    main()
