"""
Retriever evaluation for the RAG pipeline.

Measures:
  - Contextual Recall    : does the retrieved context contain the information needed to answer the question?
  - Contextual Precision : are the most relevant context chunks ranked near the top of the retrieval results?

The retriever is evaluated in isolation from generation by passing retrieved context and ideal answers to DeepEval metrics.
This ensures performance metrics reflect retrieval and reranking quality, independent of generator capabilities.

"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import CacheConfig, DisplayConfig, AsyncConfig

from src.search import RAGSearch
from src.hybrid_search import HybridRAGSearch
from src.reranker import CrossEncoderReranker

load_dotenv()

DATASET = Path("evals/goldens/golden_dataset.json")
TOP_K = 8

RECALL_THRESHOLD = 0.7
PRECISION_THRESHOLD = 0.6

JUDGE_MODEL = "gpt-4o-mini"

def load_dataset():
    with open(DATASET, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retriever(name, retriever, dataset):
    print(f"\n{'=' * 50}")
    print(name)
    print(f"{'=' * 50}")

    metrics = [
        ContextualRecallMetric(threshold=RECALL_THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=PRECISION_THRESHOLD, model=JUDGE_MODEL, include_reason=True)
    ]

    test_cases = []

    for item in dataset:
        query = item["query"]

        results = retriever.retrieve(query, top_k=TOP_K)
        retrieved_context = [result["text"] for result in results if result.get("text")]

        if not retrieved_context:
            continue

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output="Not evaluating generation",
                expected_output=item["ideal_answer"],
                retrieval_context=retrieved_context
            )
        )

    evaluate(
        test_cases=test_cases, 
        metrics=metrics, 
        async_config=AsyncConfig(max_concurrent=10), 
        display_config=DisplayConfig(results_folder="./eval-results"), 
        cache_config=CacheConfig(write_cache=False)
    )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    dataset = load_dataset()
    retrievers = {
        "Hybrid Search + Reranker": HybridRAGSearch(reranker=CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")),
        # "Vector Search": RAGSearch(),
    }

    for name, retriever in retrievers.items():
        evaluate_retriever(name, retriever, dataset)


if __name__ == "__main__":
    main()