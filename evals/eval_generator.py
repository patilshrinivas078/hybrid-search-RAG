"""
Generator evaluation for the RAG pipeline.

Measures:
  - Faithfulness     : does the generated answer stay grounded in context?
  - Answer Relevancy : does the answer actually address the question?

The generator is evaluated in isolation using GOLDEN context rather than
retriever output. This means failures are attributable to the generator,
not retrieval.

"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import CacheConfig, DisplayConfig, AsyncConfig
from src.generator import generate


load_dotenv()

GOLDEN_DATASET = Path("evals/goldens/golden_generation_eval.json")
JUDGE_MODEL = "gpt-5-mini"

THRESHOLD = 0.7

def load_dataset():
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_generator(dataset):

    print(f"\n{'=' * 50}")
    print("Evaluating Generator")
    print(f"{'=' * 50}")
    
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
    ]

    test_cases = []
    for item in dataset:
        query = item["query"]
        context = item["ideal_context"]

        answer = generate(query, context)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context
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

    evaluate_generator(dataset)


if __name__ == "__main__":
    main()