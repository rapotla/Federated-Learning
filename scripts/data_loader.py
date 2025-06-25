import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_heart_disease_data(client_id=0, num_clients=5, batch_size=32):
    # Load from UCI repository (pre-downloaded or manually placed file)
    df = pd.read_csv("data/heart.csv")  # Should be placed manually
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Shuffle and split into clients
    total_samples = len(y)
    samples_per_client = total_samples // num_clients
    start = client_id * samples_per_client
    end = (client_id + 1) * samples_per_client

    X_client = X[start:end]
    y_client = y[start:end]

    scaler = StandardScaler()
    X_client = scaler.fit_transform(X_client)

    tensor_x = torch.tensor(X_client, dtype=torch.float32)
    tensor_y = torch.tensor(y_client, dtype=torch.float32)

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_credit_data(client_id=0, num_clients=5, batch_size=32):
    # Load from Kaggle (Give Me Some Credit, manually placed file)
    df = pd.read_csv("data/credit.csv")  # Should be placed manually
    df = df.dropna()

    X = df.drop("SeriousDlqin2yrs", axis=1).values
    y = df["SeriousDlqin2yrs"].values

    total_samples = len(y)
    samples_per_client = total_samples // num_clients
    start = client_id * samples_per_client
    end = (client_id + 1) * samples_per_client

    X_client = X[start:end]
    y_client = y[start:end]

    scaler = StandardScaler()
    X_client = scaler.fit_transform(X_client)

    tensor_x = torch.tensor(X_client, dtype=torch.float32)
    tensor_y = torch.tensor(y_client, dtype=torch.float32)

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
