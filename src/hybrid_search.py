from pathlib import Path

from langfuse import observe, get_client

from src.data_loader import load_all_documents, load_document
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.embeddings import EmbeddingModel
from src.sparse_search import BM25Search

RRF_K = 60

class HybridRAGSearch:
    def __init__(self, persist_dir: str = "chroma_db", data_dir: str = "data", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5", llm_model: str = "openai/gpt-oss-20b", dense_weight: float = 0.6, sparse_weight: float = 0.4, reranker=None):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker  # None for now -- a cross-encoder slots in here later; nothing else changes

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Stored as an attribute (not created inline) so add_document() below
        # reuses the exact same chunk_size/chunk_overlap as the initial build,
        # rather than risking a second RecursiveChunker() with different defaults.
        self.chunker = RecursiveChunker()

        if self.vectorstore.collection_exists():
            self.vectorstore.load()
        else:
            docs, failures = load_all_documents("data")
            chunks = RecursiveChunker().chunk_documents(docs)
            embeddings = self.embedder.embed_chunks(chunks)
            self.vectorstore.build_from_chunks(chunks, embeddings)

        ids, texts, metadatas = self.vectorstore.get_all_chunks()
        self.bm25 = BM25Search(ids, texts, metadatas)
        print(f"[INFO] BM25 index built over {len(ids)} chunks")

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, top_k: int):
        scores: dict = {}
        lookup: dict = {}
        for rank, r in enumerate(dense_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + self.dense_weight * (1.0 / (RRF_K + rank + 1))
            lookup[r["id"]] = r
        for rank, r in enumerate(sparse_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + self.sparse_weight * (1.0 / (RRF_K + rank + 1))
            lookup.setdefault(r["id"], r)

        fused_ids = sorted(scores, key=lambda i: -scores[i])[:top_k]
        return [lookup[i] for i in fused_ids]

    @observe(name="hybrid_retrieve")
    def retrieve(self, query: str, top_k: int = 5):
        candidate_k = top_k * 4 if self.reranker else top_k

        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vectorstore.query(query_embedding, top_k=candidate_k)
        sparse_results = self.bm25.search(query, top_k=candidate_k)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k=candidate_k)

        if self.reranker is not None:
            results = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            results = fused[:top_k]
        
        
        get_client().update_current_span(
            input={"query": query, "top_k": top_k, "candidate_k": candidate_k},
            output={"retrieved_count": len(results)},
            metadata={"reranked": self.reranker is not None},
        )
        
        return results

    @observe(name="add_document")
    def add_document(self, file_path: str) -> dict:
        """
        Add a single PDF/TXT/DOCX file to the knowledge base permanently.
 
        Chroma supports this cheaply: build_from_chunks() upserts by a
        content-hashed ID, so re-adding only touches the new file's chunks -
        nothing existing gets re-embedded or disturbed.
 
        BM25 can't be updated incrementally (its term stats are computed
        over the whole corpus at construction time), so it gets rebuilt from
        everything currently in Chroma after the upsert.
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
 
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
