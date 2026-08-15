import logging
import sys
from src.hybrid_search import HybridRAGSearch
from src.reranker import CrossEncoderReranker

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", stream=sys.stdout, force=True,)

def main():
    setup_logging()
    reranker = CrossEncoderReranker()
    rag_search = HybridRAGSearch(reranker=reranker)

    print("\nRAG system ready. Type a question, or press Enter / type 'exit' to quit.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit", ""):
            break
        answer = rag_search.answer(query, top_k=3)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()