"""
Intent Classifier — Phase 3
Three-tier classifier: keyword baseline → SetFit → Groq LLM fallback
"""
import sys
import re
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.classifier.intents import INTENTS, INTENT_NAMES


# ─── Tier 1: Keyword Classifier (Baseline) ────────────────────────────────────

class KeywordClassifier:
    """
    Rule-based keyword classifier.
    Used as: (1) Baseline comparison, (2) Fast pre-filter.
    """

    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (intent, confidence).
        Confidence = number of keywords matched / total possible.
        """
        text_lower = text.lower()
        scores = {}

        for intent_name, intent_def in INTENTS.items():
            matched = sum(1 for kw in intent_def["keywords"] if kw in text_lower)
            if matched > 0:
                scores[intent_name] = matched / len(intent_def["keywords"])

        if not scores:
            return "general_inquiry", 0.3

        best_intent = max(scores, key=scores.get)
        return best_intent, min(scores[best_intent] * 3, 0.85)  # Cap at 0.85

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]


# ─── Tier 2: LLM Classifier (Groq) ───────────────────────────────────────────

class LLMClassifier:
    """
    Groq-powered intent classifier using few-shot prompting.
    Most accurate, used when keyword confidence is low.
    """

    SYSTEM_PROMPT = """You are an expert intent classifier for customer support tweets directed at {brand}.

Your task: classify the customer message into EXACTLY ONE of these intents:

{intent_list}

Rules:
1. Return ONLY the intent key (e.g., "login_problem"). Nothing else.
2. Pick the PRIMARY issue if multiple are present.
3. When unsure, use "general_inquiry".
4. Do not make up new intents."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _build_intent_list(self) -> str:
        lines = []
        for name, defn in INTENTS.items():
            lines.append(f"- {name}: {defn['description']}")
        return "\n".join(lines)

    def predict(self, text: str, retries: int = 2) -> tuple[str, float]:
        """Classify a single message. Returns (intent, confidence)."""
        client = self._get_client()

        system_prompt = self.SYSTEM_PROMPT.format(
            brand=settings.target_brand,
            intent_list=self._build_intent_list(),
        )

        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Customer message: {text}"},
                    ],
                    max_tokens=20,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip().lower()

                # Extract intent from response
                intent = raw.replace('"', "").replace("'", "").strip()
                if intent in INTENT_NAMES:
                    return intent, 0.92
                else:
                    # Try to find a match in the response
                    for name in INTENT_NAMES:
                        if name in raw:
                            return name, 0.85
                    logger.warning(f"LLM returned unexpected intent: '{raw}', defaulting to general_inquiry")
                    return "general_inquiry", 0.5

            except Exception as e:
                if attempt < retries:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"LLM classification failed: {e}")
                    return "general_inquiry", 0.3

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Batch classification with rate limiting."""
        import time
        results = []
        for i, text in enumerate(texts):
            results.append(self.predict(text))
            if i > 0 and i % 10 == 0:
                time.sleep(1)  # Respect Groq rate limits
        return results


# ─── Hybrid Classifier (production) ──────────────────────────────────────────

class HybridClassifier:
    """
    Production classifier: keyword → LLM fallback.
    Uses keyword if confidence is high, LLM otherwise.
    Saves ~80% of LLM API calls.
    """

    def __init__(self, keyword_threshold: float = 0.5):
        self.keyword_clf = KeywordClassifier()
        self.llm_clf = LLMClassifier()
        self.keyword_threshold = keyword_threshold

    def predict(self, text: str) -> dict:
        """
        Returns classification result with full metadata.
        {intent, confidence, method, escalate}
        """
        # Try keyword classifier first
        kw_intent, kw_conf = self.keyword_clf.predict(text)

        if kw_conf >= self.keyword_threshold:
            intent, confidence, method = kw_intent, kw_conf, "keyword"
        else:
            # Fall back to LLM
            llm_intent, llm_conf = self.llm_clf.predict(text)
            intent, confidence, method = llm_intent, llm_conf, "llm"

        return {
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "intent_label": INTENTS[intent]["label"],
            "intent_description": INTENTS[intent]["description"],
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# Global instance
classifier = HybridClassifier()
