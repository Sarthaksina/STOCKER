"""
Stock selection and ranking using ML models (RandomForest, LightGBM, etc.).
Scores stocks for portfolio inclusion based on financial, technical, and sentiment features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

class StockRanker:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self

# Example usage:
if __name__ == "__main__":
    # Dummy data: columns = [pe, pb, roe, momentum, sentiment, volatility]
    X = np.array([
        [18, 2, 15, 1.1, 0.7, 0.2],
        [25, 4, 8, 0.9, 0.3, 0.4],
        [12, 1.5, 20, 1.2, 0.8, 0.15],
        [30, 5, 5, 0.8, 0.2, 0.5],
        [20, 3, 12, 1.0, 0.6, 0.25]
    ])
    # Target: next quarter return (regression)
    y = np.array([0.10, -0.02, 0.15, -0.05, 0.07])
    ranker = StockRanker().fit(X, y)
    scores = ranker.predict(X)
    print("Stock scores:", scores)
    ranker.save("stock_ranker.pkl")
