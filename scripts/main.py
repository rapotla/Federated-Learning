import argparse
from models.model import MLP
from scripts.client import FederatedClient
from scripts.server import FederatedServer
from scripts.data_loader import load_heart_disease_data
from scripts.utils import plot_metrics

def main(args):
    global_model = MLP()
    clients = [FederatedClient(i) for i in range(args.num_clients)]
    test_loader = load_heart_disease_data(client_id=0)

    server = FederatedServer(global_model, clients, test_loader)
    metrics = server.train_rounds(
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        lr=args.lr,
        save_model=True,
        model_path=args.model_path
    )
    plot_metrics(metrics)
    server.evaluate_on_clients()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Runner")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--model_path", type=str, default="models/global_model.pt", help="Path to save the trained model")

    args = parser.parse_args()
    main(args)
