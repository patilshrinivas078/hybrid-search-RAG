import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Search:
    def __init__(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        self.ids = ids
        self.texts = texts
        self.metadatas = metadatas
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {"id": self.ids[i], "text": self.texts[i], "metadata": self.metadatas[i], "score": float(scores[i])}
            for i in ranked
        ]