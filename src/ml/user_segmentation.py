"""
User segmentation for portfolio analytics: KMeans clustering to group users by risk profile, investment style, or behavior.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class UserSegmenter:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path):
        joblib.dump({
            'scaler': self.scaler,
            'model': self.model
        }, path)

    def load(self, path):
        obj = joblib.load(path)
        self.scaler = obj['scaler']
        self.model = obj['model']
        return self

# Example usage:
if __name__ == "__main__":
    # Dummy data: [age, income, invest_amt, risk_score, past_return]
    X = np.array([
        [25, 40000, 10000, 8, 0.12],
        [45, 120000, 200000, 3, 0.05],
        [35, 80000, 50000, 6, 0.09],
        [28, 60000, 20000, 7, 0.11],
        [55, 200000, 500000, 2, 0.04]
    ])
    segmenter = UserSegmenter(n_clusters=3).fit(X)
    labels = segmenter.predict(X)
    print("Cluster labels:", labels)
    segmenter.save("user_segmenter.pkl")
