"""
document_indexer.py used for Knowledge Base Management

Provides the DocumentIndexer class, which is responsible for building and
maintaining the knowledge base used by the RAG pipeline.

On first run, DocumentIndexer loads all documents from the data directory,
splits them into chunks, generates embeddings, and persists them to ChromaDB.
On subsequent runs, it loads the existing ChromaDB collection directly,
skipping the expensive embed step.

Tables are handled separately from ordinary text: data_loader.py's Docling
path pulls each table out as raw HTML (no summary yet), table_store.py
summarizes it with an LLM and persists the HTML in a TableDocStore keyed by
table_id, and the resulting summary Document is merged into the same chunk
list as the regular text before embedding. Retrieval-time resolution (swap
the summary back out for the full HTML) is hybrid_search.py's job, not
this one's -- this class only builds and writes the index.

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
- table_store.py  : table HTML persistence (TableDocStore) + LLM summarization
- hybrid_search.py: queries the index (and resolves table HTML at query
                     time via TableDocStore); never writes to it
- embeddings.py   : generates the dense vector embeddings stored in Chroma
- sparse_search.py: provides the BM25 index built from Chroma's stored chunks
"""

import logging
from pathlib import Path

from langfuse import observe, get_client

from src.data_loader import load_all_documents, load_document
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.table_store import TableDocStore, build_table_documents
from src.embeddings import EmbeddingModel
from src.sparse_search import BM25Search

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(self, persist_dir: str = "chroma_db", data_dir: str = "data", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)
        # Lives alongside the Chroma index, not under data_dir: it's part of
        # the persisted index (hybrid_search.py needs it on every query, not
        # just at ingestion), so it should reload with persist_dir the same
        # way the Chroma collection does.
        self.table_store = TableDocStore(db_path=str(Path(persist_dir) / "table_store.sqlite"))
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = RecursiveChunker()

        if self.vectorstore.collection_exists():
            self.vectorstore.load()
        else:
            docs, raw_tables, failures = load_all_documents(str(self.data_dir))
            table_docs = build_table_documents(raw_tables, self.table_store)
            chunks = self.chunker.chunk_documents(docs + table_docs)
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

        loaded = load_document(dest_path)  # raises on unsupported/broken file
        table_docs = build_table_documents(loaded["tables"], self.table_store)
        new_chunks = self.chunker.chunk_documents(loaded["documents"] + table_docs)

        if not new_chunks:
            raise ValueError(f"No extractable text found in {dest_path.name}")

        new_embeddings = self.embedder.embed_chunks(new_chunks)
        self.vectorstore.build_from_chunks(new_chunks, new_embeddings)

        ids, texts, metadatas = self.vectorstore.get_all_chunks()
        self.bm25 = BM25Search(ids, texts, metadatas)

        result = {
            "filename": dest_path.name,
            "chunks_added": len(new_chunks),
            "tables_added": len(table_docs),
            "policy_type": new_chunks[0].metadata.get("policy_type", "unknown"),
            "total_chunks_in_index": len(ids),
        }
        get_client().update_current_span(input={"file": dest_path.name}, output=result)
        return result