import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import load_all_documents
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.embeddings import EmbeddingModel
from src.sparse_search import BM25Search

load_dotenv()

SYSTEM_INSTRUCTIONS = """You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't know based on the available documents" instead of guessing.
Do not use any outside knowledge."""

RRF_K = 60

class HybridRAGSearch:
    def __init__(self, persist_dir: str = "chroma_db", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5", llm_model: str = "openai/gpt-oss-20b", dense_weight: float = 0.5, sparse_weight: float = 0.5, reranker=None):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker  # None for now -- a cross-encoder slots in here later; nothing else changes

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

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to a .env file in your project root")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

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

    def retrieve(self, query: str, top_k: int = 5):
        candidate_k = top_k * 4 if self.reranker else top_k

        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vectorstore.query(query_embedding, top_k=candidate_k)
        sparse_results = self.bm25.search(query, top_k=candidate_k)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k=candidate_k)

        if self.reranker is not None:
            return self.reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]

    def answer(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        # for r in results:
        #     print("---")
        #     print(r["text"][:300])
        texts = [r["text"] for r in results if r.get("text")]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""{SYSTEM_INSTRUCTIONS}
            Context:
            {context}

            Question: {query}

            Answer:"""
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    from src.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    hybrid_search = HybridRAGSearch(reranker=reranker)
    query = "What is Self-attention mechanism?"
    answer = hybrid_search.answer(query, top_k=3)
    print("Answer:", answer)