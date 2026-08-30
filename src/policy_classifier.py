"""Keyword-based policy type classifier for document-level metadata tagging,
with an LLM fallback for documents the keyword pass can't confidently classify."""
import logging
from typing import Dict, List, Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

POLICY_KEYWORDS = {
    "two_wheeler": ["two wheeler", "two-wheeler", "motorcycle", "scooter", "pillion"],
    "car": ["private car", "motor car", "four wheeler", "windshield"],
    "home": ["householder", "home insurance", "burglary", "contents cover", "hearth"],
    "travel": ["travel insurance", "travel prime", "medical evacuation", "checked baggage", "trip cancellation"],
}

SAMPLE_CHARS = 3000

class PolicyClassification(BaseModel):
    policy_type: Optional[Literal["two_wheeler", "car", "home", "travel"]] = Field(
        description="The insurance policy type this document belongs to, or null if none clearly apply."
    )

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(PolicyClassification)


def classify_policy_type(text: str) -> Optional[str]:
    """Weighted keyword frequency match. Returns None if nothing matched."""
    sample = text[:SAMPLE_CHARS].lower()
    scores = {ptype: sum(sample.count(kw) for kw in kws) for ptype, kws in POLICY_KEYWORDS.items()}
    best_type = max(scores, key=lambda k: scores.get(k, 0))
    return best_type if scores[best_type] > 0 else None


def classify_policy_type_llm(text: str) -> Optional[str]:
    """LLM fallback classifier. Only call when keyword matching returns None."""
    try:
        result = _llm.invoke(f"Classify this insurance policy document.\n\n{text[:SAMPLE_CHARS]}")
    except Exception as e:
        logger.error("LLM policy classification failed: %s", e)
        return None
    if isinstance(result, dict):
        return result.get("policy_type")
    return result.policy_type


def classify_document_policy_type(text: str) -> Optional[str]:
    """Keyword pass first. LLM fallback only if inconclusive."""
    policy_type = classify_policy_type(text)
    if policy_type is not None:
        return policy_type
    logger.info("Keyword classification inconclusive, falling back to LLM")
    return classify_policy_type_llm(text)