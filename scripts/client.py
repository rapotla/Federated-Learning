import torch
from models.model import MLP
from scripts.data_loader import load_heart_disease_data  # Change to `load_credit_data` for credit dataset

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.local_model = MLP()
        self.train_loader = load_heart_disease_data(client_id)  # Can be swapped for load_credit_data

    def set_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())

    def train(self, epochs=1, lr=0.01):
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        self.local_model.train()
        for _ in range(epochs):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.local_model(x).squeeze()
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

    def get_weights(self):
        return list(self.local_model.state_dict().values())
