import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def get_dataloaders(client_id, batch_size=32):
    np.random.seed(client_id)
    X = np.random.randn(200, 13)
    y = (np.random.rand(200) > 0.5).astype(int)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size), None

def evaluate_model(model):
    X = torch.randn(200, 13)
    y = (torch.rand(200) > 0.5).int()
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    preds_label = (preds > 0.5).astype(int)
    return accuracy_score(y, preds_label), roc_auc_score(y, preds)

def add_noise(weights, epsilon=0.1):
    return [w + epsilon * torch.randn_like(w) for w in weights]
