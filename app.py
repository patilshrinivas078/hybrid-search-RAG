import logging
import sys
from langfuse import observe, get_client
from src.hybrid_search import HybridRAGSearch
from src.reranker import CrossEncoderReranker
from src.generator import generate

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", stream=sys.stdout, force=True,)

@observe(name="handle_query")
def handle_query(retriever, query: str, top_k: int = 4) -> str:
    results = retriever.retrieve(query, top_k=top_k)
    texts = [result["text"] for result in results if result.get("text")]
    answer = generate(query, texts)

    # Set the trace-level input/output so the Langfuse dashboard shows the
    # question -> final answer directly on the trace, not just on a nested
    # span. Everything retrieve() and generate() log internally nests
    # under this same trace automatically since they're called from here.
    get_client().update_current_span(
        input=query,
        output=answer,
        metadata={"top_k": top_k, "context_chunks": len(texts)},
    )
    return answer

def main():
    setup_logging()
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device="cuda")
    retriever = HybridRAGSearch(reranker=reranker)

    print("\nRAG system ready. Type a question, or press Enter / type 'exit' to quit.\n")
    try:
        while True:
            query = input("Question: ").strip()
            if query.lower() in ("exit", "quit", ""):
                break
            answer = handle_query(retriever, query, top_k=4)
            print(f"\nAnswer: {answer}\n")
    finally:
        # CLI scripts exit as soon as the loop breaks, unlike a server that
        # stays alive - flush explicitly so any buffered traces actually
        # get sent to Langfuse instead of being dropped on exit.
        get_client().flush()

if __name__ == "__main__":
    main()