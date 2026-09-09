"""
Reply Drafter — Phase 3
RAG-grounded reply generation using Groq LLM.
Retrieves similar historical cases, then drafts a contextually appropriate reply.
"""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.rag.vector_store import vector_store
from src.classifier.intents import INTENTS


DRAFT_SYSTEM_PROMPT = """You are a helpful, professional customer support agent for {brand}.

Your tone: Empathetic, concise, action-oriented. Similar to how {brand} actually responds on Twitter.
Your constraints:
- Maximum 280 characters (Twitter limit)
- Always acknowledge the customer's issue first
- Provide a concrete next step when possible
- Never promise something you can't deliver
- Never be defensive or dismissive

You will be given:
1. The customer's message
2. Examples of how {brand} has responded to SIMILAR issues in the past (for style/tone reference)
3. The classified intent (for context)

Use the examples as style guides, not scripts. Draft a fresh, contextually appropriate reply."""

DRAFT_USER_PROMPT = """Customer message: {customer_message}

Intent: {intent_label}

Similar historical responses (use as style reference only):
{examples}

Draft a reply to this customer message. Return ONLY the reply text, nothing else."""


class ReplyDrafter:
    """
    RAG-grounded reply drafter.
    Retrieves similar cases from vector store, feeds to Groq to draft reply.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _format_examples(self, retrieved: list[dict]) -> str:
        """Format retrieved examples for the prompt."""
        if not retrieved:
            return "No similar examples found — use your best judgment."

        lines = []
        for i, r in enumerate(retrieved[:3]):  # Max 3 examples to stay in context
            lines.append(
                f"Example {i+1} (similarity: {r['score']:.2f}):\n"
                f"  Customer: {r['customer_message'][:100]}\n"
                f"  {settings.target_brand}: {r['brand_reply'][:150]}"
            )
        return "\n\n".join(lines)

    def draft(
        self,
        customer_message: str,
        intent: str,
        retrieved_examples: Optional[list[dict]] = None,
        retries: int = 2,
    ) -> dict:
        """
        Draft a reply to a customer message.

        Args:
            customer_message: The customer's tweet
            intent: Classified intent key
            retrieved_examples: Pre-retrieved similar cases (if None, retrieves automatically)
            retries: Number of retries on failure

        Returns:
            {draft, retrieved_count, avg_retrieval_score, fallback_used}
        """
        # Retrieve similar cases if not provided
        if retrieved_examples is None:
            retrieved_examples = vector_store.search(customer_message)

        intent_info = INTENTS.get(intent, INTENTS["general_inquiry"])
        examples_text = self._format_examples(retrieved_examples)

        client = self._get_client()

        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": DRAFT_SYSTEM_PROMPT.format(brand=settings.target_brand),
                        },
                        {
                            "role": "user",
                            "content": DRAFT_USER_PROMPT.format(
                                customer_message=customer_message,
                                intent_label=intent_info["label"],
                                examples=examples_text,
                            ),
                        },
                    ],
                    max_tokens=100,
                    temperature=0.7,
                )

                draft = response.choices[0].message.content.strip()
                # Trim to Twitter limit if needed
                if len(draft) > 280:
                    draft = draft[:277] + "..."

                avg_score = (
                    sum(r["score"] for r in retrieved_examples) / len(retrieved_examples)
                    if retrieved_examples else 0.0
                )

                return {
                    "draft": draft,
                    "retrieved_count": len(retrieved_examples),
                    "avg_retrieval_score": round(avg_score, 3),
                    "fallback_used": False,
                }

            except Exception as e:
                if attempt < retries:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Reply drafting failed: {e}")
                    # Return template fallback
                    template = intent_info.get("auto_reply_template", "Thank you for reaching out! We're looking into this and will get back to you shortly.")
                    return {
                        "draft": template[:280],
                        "retrieved_count": 0,
                        "avg_retrieval_score": 0.0,
                        "fallback_used": True,
                    }


# Global instance
drafter = ReplyDrafter()
