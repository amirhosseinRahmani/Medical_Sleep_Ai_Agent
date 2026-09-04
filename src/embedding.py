import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Small wrapper around the sentence-transformers embedding model."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts):
        """Convert a list of texts into normalized float32 vectors."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")

    def create_query_embedding(self, question):
        """Convert one user question into one vector."""
        embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")
