# Federated Learning for Privacy-Preserving AI

This repository contains the implementation of our MDPI Q1 journal paper:

**"Federated Learning for Privacy-Preserving AI: A Scalable Approach to Decentralized Model Training in Healthcare and Finance"**

## 📘 Overview

This project implements a federated learning framework using PyTorch, simulating decentralized training with privacy-preserving aggregation techniques. The architecture supports multiple clients with non-IID data, adaptive aggregation, and differential privacy.

## 🛠 Features

- Federated averaging across simulated clients
- Adaptive noise-based privacy mechanism
- Modular client/server model
- Evaluation using accuracy and AUC
- Custom MLP model for tabular data

## 📂 Project Structure

```
├── scripts/main.py             # Entrypoint script
├── scripts/server.py           # FederatedServer class
├── scripts/client.py           # FederatedClient class
├── models/model.py            # MLP architecture
├── scripts/utils.py            # Dataset simulation, evaluation, privacy utilities
└── requirements.txt    # Environment dependencies
```

## 🚀 How to Run

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training:
   ```bash
   python main.py
   ```

## 📊 Datasets

While simulated data is used in this repo, the paper evaluates on:

- UCI Heart Disease: [Link](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Give Me Some Credit: [Link](https://www.kaggle.com/c/GiveMeSomeCredit/data)

## 🔐 Privacy

Noise is added to local model weights before aggregation using a differential privacy-inspired mechanism (Gaussian noise).

## 📜 License

MIT License. Free to use and modify for research purposes.
