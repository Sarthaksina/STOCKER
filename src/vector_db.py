import os
import json
import logging
from typing import List, Tuple

import numpy as np
import faiss
from src.configuration.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Ensure data directory exists
DATA_DIR = settings.data_dir
os.makedirs(DATA_DIR, exist_ok=True)

# Paths for FAISS index and ID mapping file
INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
IDS_PATH = os.path.join(DATA_DIR, 'faiss_ids.json')

class FaissDB:
    """
    Simple FAISS-based vector store using L2 distance.
    Stores vectors and their corresponding IDs.
    """
    def __init__(self, dim: int):
        self.dim = dim
        # Load existing index and IDs if available
        if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(IDS_PATH, 'r', encoding='utf-8') as f:
                self.ids = json.load(f)
            logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors).")
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.ids = []
            logger.info(f"Created new FAISS index (dim={dim}).")

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors and their IDs to the index and persist."""
        # Ensure correct shape and dtype
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, \
            f"Expected vectors shape (n, {self.dim}), got {vectors.shape}"
        self.index.add(vectors.astype('float32'))
        self.ids.extend(ids)
        self._save()

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search query vector and return top-k (id, distance)."""
        assert query.ndim == 1 and query.size == self.dim, \
            f"Expected query shape ({self.dim},), got {query.shape}"
        D, I = self.index.search(query.reshape(1, -1).astype('float32'), k)
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.ids):
                results.append((self.ids[idx], float(dist)))
        return results

    def _save(self) -> None:
        """Persist index and ID list to disk."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.ids, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors). IDs saved: {len(self.ids)}.")

# Convenience function to get a singleton FaissDB instance
_faiss_db: FaissDB = None

def get_faiss_db(dim: int) -> FaissDB:
    global _faiss_db
    if _faiss_db is None:
        _faiss_db = FaissDB(dim)
    return _faiss_db
