import logging
from typing import Any, Dict, List, Optional, cast
from langfuse import observe
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        logger.info("Loading cross-encoder reranker: %s (device=%s)", model_name, device or "auto")
        print("[RERANKER] Starting model load...")
        self.model = CrossEncoder(model_name, device=device)
        print("[RERANKER] Model loaded successfully!")
        logger.info("Cross-encoder reranker loaded successfully")

    @observe(name="rerank")
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, str(c["text"])) for c in candidates]
        scores = self.model.predict(cast(Any, pairs))

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda c: -c["rerank_score"])
        return reranked[:top_k]
 