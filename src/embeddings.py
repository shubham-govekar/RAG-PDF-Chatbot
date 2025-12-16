import functools
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import config

class EmbeddingService:
    def __init__(self):
        self.model = None

    def load_model(self) -> None:
        """Lazy load the SentenceTransformer model."""
        if not self.model:
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Batch embed a list of texts."""
        self.load_model()
        return self.model.encode(
            texts, 
            batch_size=config.EMBEDDING_BATCH_SIZE, 
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )

    @functools.lru_cache(maxsize=100)
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string with caching."""
        self.load_model()
        return self.model.encode(query, normalize_embeddings=True)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata matching app.py expectations."""
        self.load_model()
        return {
            "dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "loaded": True  # Required by app.py line 128
        }

_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service