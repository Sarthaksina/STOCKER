"""
Neural network model to predict user risk tolerance or recommend asset allocation based on user profile/history.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

class UserAllocationNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# --- Training function ---
def train_nn(X, y, input_dim, output_dim, epochs=100, lr=0.01):
    model = UserAllocationNN(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss() if output_dim > 1 else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long if output_dim > 1 else torch.float32)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# --- Inference function ---
def predict_allocation(model, user_features):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(user_features, dtype=torch.float32)
        output = model(X_tensor)
        return output.numpy()

# --- Example usage ---
if __name__ == "__main__":
    # Dummy data: [age, income, invest_amt, risk_score, past_return]
    X = np.array([
        [25, 40000, 10000, 8, 0.12],
        [45, 120000, 200000, 3, 0.05],
        [35, 80000, 50000, 6, 0.09],
        [28, 60000, 20000, 7, 0.11],
        [55, 200000, 500000, 2, 0.04]
    ])
    # Asset allocation classes: 0=Aggressive, 1=Moderate, 2=Conservative
    y = np.array([0, 2, 1, 0, 2])
    model = train_nn(X, y, input_dim=5, output_dim=3, epochs=200)
    test_user = np.array([[30, 70000, 25000, 7, 0.10]])
    allocation_probs = predict_allocation(model, test_user)
    print("Allocation class probabilities:", allocation_probs)
    # Save model
    torch.save(model.state_dict(), "user_allocation_nn.pt")
