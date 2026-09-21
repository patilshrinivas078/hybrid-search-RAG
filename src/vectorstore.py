import os
import logging
from typing import Any, Dict, List
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
        # Table chunks already carry a stable, unique table_id (assigned in
        # data_loader.py, reused as-is in table_store.py) that's also the
        # TableDocStore key. Reuse it directly so the vector store id and
        # the docstore key can never drift apart -- that link is what makes
        # resolve_full_content() below work.
        table_id = chunk.metadata.get("table_id")
        if table_id:
            return f"table::{table_id}"

        source = chunk.metadata.get("source", "unknown")
        text_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
        return f"{source}::{text_hash}"

    def build_from_chunks(self, chunks: List[Any], embeddings):
        logger.info(f"Building vector store from {len(chunks)} chunks...")

        self.collection = self.client.get_or_create_collection(name="policy_documents")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [self._chunk_id(chunk) for chunk in chunks] #same chunks will always get same IDs, eventually being overwritten.

        # Keep the first occurrence of each id and drop the rest, storing identical 
        # text under a second id adds no retrievable information anyway.
        embeddings_list = embeddings.tolist()
        seen_ids = set()
        dedup_ids, dedup_texts, dedup_embeddings, dedup_metadatas = [], [], [], []
        for chunk_id, text, emb, meta in zip(ids, texts, embeddings_list, metadatas):
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            dedup_ids.append(chunk_id)
            dedup_texts.append(text)
            dedup_embeddings.append(emb)
            dedup_metadatas.append(meta)

        n_dropped = len(ids) - len(dedup_ids)
        if n_dropped:
            logger.warning(
                "Dropped %d duplicate-id chunk(s) within this batch (identical "
                "page_content hashing to the same id) before upsert", n_dropped,
            )

        # .upsert -> if ID already exists in the collection, it overwrites that record; if it doesn't exist yet, it inserts a new one.
        self.collection.upsert(ids=dedup_ids, documents=dedup_texts, embeddings=dedup_embeddings, metadatas=dedup_metadatas)
        logger.info(f"Added/updated {len(dedup_ids)} chunks in ChromaDB")

    def collection_exists(self) -> bool:
        """Check whether the collection has already been built"""
        try:
            self.client.get_collection(name="policy_documents")
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
        self.collection = self.client.get_collection(name="policy_documents")
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

    def resolve_full_content(self, results: List[Dict[str, Any]], table_store: Any) -> List[Dict[str, Any]]:
        """
        The multi-vector swap-in step. `results` came back from query(),
        matched by similarity against whatever was embedded -- for table
        chunks, that's the LLM-written summary, not the table itself. For
        any result whose metadata has a table_id, replace "text" with the
        full HTML pulled from table_store (a TableDocStore, or anything
        with a compatible .get(table_id) -> {"html": ...}) so the
        generation LLM sees the actual table structure instead of the
        summary it was matched on. Non-table results pass through untouched.

        Call this on the results of query() before building the generation
        prompt, e.g.:
            results = vectorstore.query(query_embedding, top_k=5)
            results = vectorstore.resolve_full_content(results, table_store)
        """
        resolved = []
        for r in results:
            r = dict(r)
            table_id = r["metadata"].get("table_id")
            if table_id:
                record = table_store.get(table_id)
                if record:
                    r["summary"] = r["text"]  # kept for citation/debugging
                    r["text"] = record["html"]
                else:
                    logger.warning("table_id '%s' in metadata but missing from table_store", table_id)
            resolved.append(r)
        return resolved