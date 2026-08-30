import os
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.embeddings import EmbeddingModel

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "chroma_db", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)

        if self.vectorstore.collection_exists():
            self.vectorstore.load()
        else:
            docs, failures = load_all_documents("data")
            chunks = RecursiveChunker().chunk_documents(docs)
            embeddings = self.embedder.embed_chunks(chunks)
            self.vectorstore.build_from_chunks(chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_query(query)
        return self.vectorstore.query(query_embedding, top_k=top_k)