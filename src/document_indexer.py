"""
document_indexer.py used for Knowledge Base Management

Provides the DocumentIndexer class, which is responsible for building and
maintaining the knowledge base used by the RAG pipeline.

On first run, DocumentIndexer loads all documents from the data directory,
splits them into chunks, generates embeddings, and persists them to ChromaDB.
On subsequent runs, it loads the existing ChromaDB collection directly,
skipping the expensive embed step.

A BM25 sparse index is also built over all stored chunks at startup. This
powers the keyword-based half of hybrid retrieval (see hybrid_search.py).

Usage
-----
    from src.document_indexer import DocumentIndexer

    indexer = DocumentIndexer(
        persist_dir="chroma_db",   # where ChromaDB is saved on disk
        data_dir="data",           # folder containing your PDF/TXT/DOCX files
    )

    # Add a new document at runtime (updates both Chroma and BM25):
    result = indexer.add_document("path/to/new_policy.pdf")

Relationship to other modules
------------------------------
- vectorstore.py  : low-level ChromaDB wrapper (querying and storage)
- hybrid_search.py: queries the index via DocumentIndexer; never writes to it
- embeddings.py   : generates the dense vector embeddings stored in Chroma
- sparse_search.py: provides the BM25 index built from Chroma's stored chunks
"""

import logging
from pathlib import Path

from langfuse import observe, get_client

from src.data_loader import load_all_documents, load_document
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.embeddings import EmbeddingModel
from src.sparse_search import BM25Search

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, persist_dir: str = "chroma_db", data_dir: str = "data", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = RecursiveChunker()

        if self.vectorstore.collection_exists():
            self.vectorstore.load()
        else:
            docs, failures = load_all_documents(str(self.data_dir))
            chunks = self.chunker.chunk_documents(docs)
            embeddings = self.embedder.embed_chunks(chunks)
            self.vectorstore.build_from_chunks(chunks, embeddings)

        ids, texts, metadatas = self.vectorstore.get_all_chunks()
        self.bm25 = BM25Search(ids, texts, metadatas)
        logger.info("BM25 index built over %d chunks", len(ids))

    @observe(name="add_document")
    def add_document(self, file_path: str) -> dict:
        """
        Add a single PDF/TXT/DOCX file to the knowledge base permanently.

        BM25 can't be updated incrementally (its term stats are computed
        over the whole corpus at construction time), so it gets rebuilt from
        everything currently in Chroma after the upsert.
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Copy into data_dir so it's part of the on-disk corpus and gets picked up automatically if the index is ever fully rebuilt from scratch (e.g. persist_dir is deleted), not just upserted now.
        dest_path = self.data_dir / source_path.name
        if source_path.resolve() != dest_path.resolve():
            dest_path.write_bytes(source_path.read_bytes())

        loaded_docs = load_document(dest_path)  # raises on unsupported/broken file
        new_chunks = self.chunker.chunk_documents(loaded_docs)

        if not new_chunks:
            raise ValueError(f"No extractable text found in {dest_path.name}")

        new_embeddings = self.embedder.embed_chunks(new_chunks)
        self.vectorstore.build_from_chunks(new_chunks, new_embeddings)

        ids, texts, metadatas = self.vectorstore.get_all_chunks()
        self.bm25 = BM25Search(ids, texts, metadatas)

        result = {
            "filename": dest_path.name,
            "chunks_added": len(new_chunks),
            "policy_type": new_chunks[0].metadata.get("policy_type", "unknown"),
            "total_chunks_in_index": len(ids),
        }
        get_client().update_current_span(input={"file": dest_path.name}, output=result)
        return result