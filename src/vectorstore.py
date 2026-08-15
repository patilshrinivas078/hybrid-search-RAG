import os
import logging
from typing import Any, List
import hashlib
import chromadb

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = None

    def _chunk_id(self, chunk) -> str:
        source = chunk.metadata.get("source", "unknown")
        text_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        return f"{source}::{text_hash}"

    def build_from_chunks(self, chunks: List[Any], embeddings):
        logger.info(f"Building vector store from {len(chunks)} chunks...")

        self.collection = self.client.get_or_create_collection(name="documents")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [self._chunk_id(chunk) for chunk in chunks] #same chunks will always get same IDs, eventually being overwritten.

        # .upsert -> if ID already exists in the collection, it overwrites that record; if it doesn't exist yet, it inserts a new one.
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas)
        logger.info(f"Added/updated {len(chunks)} chunks in ChromaDB")
        logger.info(f"Vector store built and saved to {self.persist_dir}")

    def collection_exists(self) -> bool:
        """Check whether the collection has already been built"""
        try:
            self.client.get_collection(name="documents")
            return True
        except Exception:
            return False

    def get_all_chunks(self):
        """Return everything currently stored -- used to build the BM25 index."""
        if self.collection is None:
            raise ValueError("No collection loaded. Call build_from_chunks() or load() first.")
        data = self.collection.get(include=["documents", "metadatas"])
        return data["ids"], data["documents"], data["metadatas"]

    def load(self):
        self.collection = self.client.get_collection(name="documents")
        logger.info(f"Loaded ChromaDB collection with {self.collection.count()} documents")

    def query(self, query_embedding, top_k: int = 5):
        if self.collection is None:
            raise ValueError("No collection loaded. Call build_from_chunks() or load() first.")

        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        formatted_results = []

        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return formatted_results

if __name__ == "__main__":
    from src.data_loader import load_all_documents
    from src.chunking import RecursiveChunker
    from src.embeddings import EmbeddingModel

    docs, failures = load_all_documents("data")
    chunker = RecursiveChunker()
    chunks = chunker.chunk_documents(docs)

    embedder = EmbeddingModel()
    embeddings = embedder.embed_chunks(chunks)

    store = ChromaVectorStore("chroma_db")
    store.build_from_chunks(chunks, embeddings)

    store.load()

    query = "What is Self-attention mechanism?"
    query_embedding = embedder.embed_query(query)

    results = store.query(query_embedding, top_k=3)

    print(results)