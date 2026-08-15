import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents
from src.chunking import RecursiveChunker
from src.vectorstore import ChromaVectorStore
from src.embeddings import EmbeddingModel

            
load_dotenv()

SYSTEM_INSTRUCTIONS = """You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't know based on the available documents" instead of guessing.
Do not use any outside knowledge."""

class RAGSearch:
    def __init__(self, persist_dir: str = "chroma_db", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5", llm_model: str = "openai/gpt-oss-20b"):
        self.embedder = EmbeddingModel(embedding_model)
        self.vectorstore = ChromaVectorStore(persist_dir)

        if self.vectorstore.collection_exists():
            self.vectorstore.load()
        else:
            docs, failures = load_all_documents("data")
            chunks = RecursiveChunker().chunk_documents(docs)
            embeddings = self.embedder.embed_chunks(chunks)
            self.vectorstore.build_from_chunks(chunks, embeddings)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to a .env file in your project root")
        
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_query(query)
        return self.vectorstore.query(query_embedding, top_k=top_k)

    def answer(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
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
    rag_search = RAGSearch()
    query = "What is Self-attention mechanism?"
    answer = rag_search.answer(query, top_k=3)
    print("Answer:", answer)