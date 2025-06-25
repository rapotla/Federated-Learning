import torch
from scripts.utils import evaluate_model
from collections import OrderedDict
import os
import csv

class FederatedServer:
    def __init__(self, global_model, clients, test_loader):
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader

    def aggregate_weights(self, client_weights):
        avg_weights = OrderedDict()
        for key in client_weights[0].keys():
            avg_weights[key] = torch.stack([weights[key] for weights in client_weights], dim=0).mean(dim=0)
        return avg_weights

    def train_rounds(self, num_rounds=5, local_epochs=1, lr=0.01, save_model=True, model_path="models/global_model.pt"):
        from scripts.utils import init_metrics
        metrics = init_metrics()
        log_path = "logs/metrics.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Accuracy", "AUC", "F1"])

            for rnd in range(num_rounds):
                client_weights = []
                for client in self.clients:
                    client.set_model(self.global_model)
                    client.train(epochs=local_epochs, lr=lr)
                    weights = dict(zip(self.global_model.state_dict().keys(), client.get_weights()))
                    client_weights.append(weights)

                avg_weights = self.aggregate_weights(client_weights)
                self.global_model.load_state_dict(avg_weights)

                acc, auc, f1 = evaluate_model(self.global_model, self.test_loader)
                metrics["accuracy"].append(acc)
                metrics["auc"].append(auc)
                metrics["f1"].append(f1)
                writer.writerow([rnd + 1, f"{acc:.4f}", f"{auc:.4f}", f"{f1:.4f}"])
                print(f"Round {rnd+1}/{num_rounds} — Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

        if save_model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.global_model.state_dict(), model_path)
            print(f"✅ Saved global model to {model_path}")

        return metrics

    def evaluate_on_clients(self, log_path="logs/client_eval.csv"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Client ID", "Accuracy", "AUC", "F1"])

            for client in self.clients:
                client.set_model(self.global_model)
                acc, auc, f1 = evaluate_model(self.global_model, client.data_loader)
                writer.writerow([client.client_id, f"{acc:.4f}", f"{auc:.4f}", f"{f1:.4f}"])
                print(f"📊 Client {client.client_id} — Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
