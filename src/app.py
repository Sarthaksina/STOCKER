"""
STOCKER CLI Demo App
Runs all major workflows of the agentic analytics platform with mock/demo data.
"""
import numpy as np
import pandas as pd
from src.features.agent import Agent
from src.features.modeling import (
    segment_users_kmeans, segment_users_gmm, classify_risk_profile_rf,
    build_risk_tolerance_nn, stock_ranker_rf, build_lstm_forecaster,
    analyze_sentiment_transformer, detect_anomalies_isolation_forest,
    build_autoencoder, recommend_stocks_collab
)
from src.features.news_agent import search_news
from src.features.concall_agent import analyze_concall
from src.utils.helpers import top_n_recommender
from src.features.vector_search import add_vector_similarity
from src.features.portfolio.portfolio_optimization import mean_variance_portfolio
from src.features.portfolio.portfolio_risk import exposure_analysis
from src.features.reporting import generate_report

# --- Mock Data ---
users = np.array([
    [25, 5, 1],  # [age, years investing, risk appetite]
    [40, 15, 0],
    [30, 10, 2],
])
user_labels = np.array([2, 0, 1])  # Aggressive, Conservative, Moderate

returns = pd.DataFrame(np.random.randn(100, 3), columns=["TCS", "INFY", "RELIANCE"])
portfolio = {"TCS": 0.5, "INFY": 0.3, "RELIANCE": 0.2}
sector_map = {"TCS": "IT", "INFY": "IT", "RELIANCE": "Energy"}
asset_map = {"TCS": "Equity", "INFY": "Equity", "RELIANCE": "Equity"}

news_demo = search_news("TCS")
concall_demo = analyze_concall("""We are confident about our growth. Our guidance for next quarter is strong. There are some risks, but overall the outlook is positive.""")

# --- Agent Demo ---
agent = Agent()

print("\n=== Holdings Analytics ===")
holdings_df = pd.DataFrame({"user_id": [1, 1, 2], "value": [100, 200, 300]})
print(agent.handle_query("holdings", {"df": holdings_df}))

print("\n=== Portfolio Optimization ===")
print(agent.handle_query("portfolio optimize", {"returns": returns}))

print("\n=== Exposure Analysis ===")
print(agent.handle_query("exposure", {"portfolio": portfolio, "sector_map": sector_map, "asset_map": asset_map}))

print("\n=== Reporting ===")
print(agent.handle_query("report", {"df": returns}))

print("\n=== Peer Comparison ===")
df_peer = pd.DataFrame({"score": [0.9, 0.7, 0.85]}, index=["TCS", "INFY", "RELIANCE"])
print(agent.handle_query("peer compare", {"df": df_peer, "score_col": "score", "n": 2}))

print("\n=== Vector Search ===")
df_vec = pd.DataFrame({"embedding": [np.random.rand(4) for _ in range(3)]}, index=["TCS", "INFY", "RELIANCE"])
query_vec = np.random.rand(4)
print(agent.handle_query("vector search", {"df": df_vec, "vector_col": "embedding", "query_vec": query_vec, "out_col": "similarity"}))

print("\n=== News Agent ===")
print(news_demo)

print("\n=== Concall Agent ===")
print(concall_demo)

print("\n=== ML/DL User Segmentation & Risk Profiling ===")
print("KMeans Segmentation:", segment_users_kmeans(users))
print("GMM Segmentation:", segment_users_gmm(users))
print("Random Forest Risk Classifier:", classify_risk_profile_rf(users, user_labels))

print("\n=== DL Risk Tolerance NN ===")
model = build_risk_tolerance_nn(input_dim=3)
print(model.summary())

print("\n=== Stock Ranking (RF) ===")
X_stock = np.random.rand(10, 5)
y_stock = np.random.rand(10)
rf_model = stock_ranker_rf(X_stock, y_stock)
print(rf_model)

print("\n=== LSTM Price Forecaster (DL) ===")
lstm_model = build_lstm_forecaster(input_shape=(10, 5))
print(lstm_model.summary())

print("\n=== Transformer Sentiment Analysis ===")
texts = ["TCS posts strong growth", "INFY faces negative outlook"]
print(analyze_sentiment_transformer(texts))

print("\n=== Anomaly Detection (Isolation Forest) ===")
X_anom = np.random.rand(20, 3)
print(detect_anomalies_isolation_forest(X_anom))

print("\n=== Autoencoder for Anomaly Detection ===")
autoencoder = build_autoencoder(input_dim=3)
print(autoencoder.summary())

print("\n=== Collaborative Filtering Recommendations ===")
user_item = np.array([[5, 0, 3], [0, 4, 0], [2, 0, 0]])
print(recommend_stocks_collab(user_item, user_index=0, n_recs=2))

print("\n=== Peer Comparison (Direct) ===")
print(top_n_recommender(df_peer, score_col="score", n=2))

print("\n=== Vector Similarity (Direct) ===")
print(add_vector_similarity(df_vec, vector_col="embedding", query_vec=query_vec, out_col="similarity"))

print("\n=== Portfolio Optimization (Direct) ===")
print(mean_variance_portfolio(returns))

print("\n=== Exposure Analysis (Direct) ===")
print(exposure_analysis(portfolio, sector_map, asset_map))

print("\n=== Reporting (Direct) ===")
print(generate_report(returns))
