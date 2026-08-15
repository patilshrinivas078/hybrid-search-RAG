import logging
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info("Embedding model loaded successfully")

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        logger.info(f"Generated chunk embeddings with shape: {embeddings.shape}")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        logger.info(f"Generating embedding for query: {query}")

        embedding = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)

        logger.info(f"Generated query embedding with shape: {embedding.shape}")

        return embedding[0]


if __name__ == "__main__":
    from src.data_loader import load_all_documents
    from src.chunking import RecursiveChunker

    docs, failures = load_all_documents("data")
    chunks = RecursiveChunker().chunk_documents(docs)

    embedder = EmbeddingModel()

    chunk_embeddings = embedder.embed_chunks(chunks)
    query_embedding = embedder.embed_query("What is attention mechanism?")

    print("Chunk embeddings shape:", chunk_embeddings.shape)
    print("Query embedding shape:", query_embedding.shape)