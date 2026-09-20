"""
RAG pipeline evaluation.
 
Measures:
  - Faithfulness          : does the generated answer stay grounded in the retrieved context?
  - Answer Relevancy      : does the answer actually address the question?
  - Contextual Relevancy  : is the context the retriever pulled back actually relevant to the question?
 
This evaluates the full pipeline (retriever -> generator) against a query
set, using whatever context the retriever actually returns at query time.
 
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import CacheConfig, DisplayConfig, AsyncConfig
from src.generator import generate
from src.hybrid_search import HybridRAGSearch
from src.reranker import CrossEncoderReranker


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_DATASET = Path("evals/goldens/generate_golden.json")
JUDGE_MODEL = "gpt-5-mini"

TOP_K = 5
THRESHOLD = 0.7

def load_dataset():
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_rag_pipeline(dataset, retriever):

    print(f"\n{'=' * 50}")
    print("Evaluating RAG Pipeline")
    print(f"{'=' * 50}")
    
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    test_cases = []
    for item in dataset[:10]:
        query = item["query"]

        results = retriever.retrieve(query, top_k=TOP_K)
        retrieved_context = [result["text"] for result in results if result.get("text")]

        if not retrieved_context:
            logger.warning("Skipping %s: retriever returned no context for query %r", item.get("id", query), query)
            continue

        answer = generate(query, retrieved_context)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieved_context
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=3), 
        display_config=DisplayConfig(results_folder="./eval-results"), 
        cache_config=CacheConfig(write_cache=False)
    )


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    dataset = load_dataset()
    retriever = HybridRAGSearch(reranker=CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3"))
    evaluate_rag_pipeline(dataset, retriever)

if __name__ == "__main__":
    main()