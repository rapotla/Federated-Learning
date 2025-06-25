import torch
from models.model import MLP
from scripts.data_loader import load_heart_disease_data
from scripts.utils import evaluate_model
import argparse
import os

def main(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = load_heart_disease_data(client_id=0)
    acc, auc, f1 = evaluate_model(model, test_loader)

    print(f"✅ Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Federated Model")
    parser.add_argument("--model_path", type=str, default="models/global_model.pt", help="Path to saved model file")
    args = parser.parse_args()
    main(args.model_path)
