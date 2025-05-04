"""
Stock price/return forecasting using LSTM (deep learning for time series).
Predicts future price trends for stock selection and portfolio analytics.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

# --- Training function ---
def train_lstm(prices, seq_length=10, epochs=50, lr=0.001):
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()
    X, y = create_sequences(prices_scaled, seq_length)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    model = StockLSTM(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model, scaler

# --- Inference function ---
def predict_lstm(model, scaler, recent_prices, seq_length=10):
    model.eval()
    prices_scaled = scaler.transform(np.array(recent_prices).reshape(-1, 1)).flatten()
    X_input = torch.tensor(prices_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred_scaled = model(X_input).item()
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred_price

# --- Example usage ---
if __name__ == "__main__":
    # Dummy price data
    prices = np.linspace(100, 150, 120) + np.random.normal(0, 2, 120)
    model, scaler = train_lstm(prices, seq_length=10, epochs=30)
    next_price = predict_lstm(model, scaler, prices[-10:], seq_length=10)
    print("Predicted next price:", next_price)
    # Save model
    torch.save(model.state_dict(), "stock_lstm.pt")
