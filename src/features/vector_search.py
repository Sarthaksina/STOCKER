"""
Vector search and similarity utilities for STOCKER.
- Cosine similarity, vector ranking, etc.
"""
import pandas as pd
import numpy as np

def add_vector_similarity(df: pd.DataFrame, vector_col: str = "embedding", query_vec: np.ndarray = None, out_col: str = "similarity") -> pd.DataFrame:
    """
    Adds cosine similarity to a query vector if vector_col exists.
    """
    if vector_col in df.columns and query_vec is not None:
        from numpy.linalg import norm
        def cosine_sim(v):
            v = np.array(v)
            return np.dot(v, query_vec) / (norm(v) * norm(query_vec))
        df[out_col] = df[vector_col].apply(cosine_sim)
    return df
