"""
Escalation Router — Phase 3
Decides whether a message should be auto-handled or escalated to a human.
Uses rule-based logic + LLM scoring.
"""
import sys
import re
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.classifier.intents import ALWAYS_ESCALATE


# Keywords that trigger immediate escalation
ESCALATION_TRIGGERS = {
    "legal_threat": [
        "lawyer", "attorney", "sue", "lawsuit", "legal action", "court",
        "consumer protection", "regulatory", "fcc", "ftc", "fraud",
    ],
    "severe_frustration": [
        "unacceptable", "disgusting", "worst company", "never again",
        "cancel my account", "closing my account", "report you",
        "social media", "going viral", "news station",
    ],
    "safety_concern": [
        "harm myself", "hurt myself", "suicide", "emergency",
    ],
    "data_breach": [
        "data breach", "hacked", "unauthorized access", "someone accessed",
        "identity theft", "data stolen",
    ],
}

ESCALATION_REASONS = {
    "legal_threat": "Customer mentioned legal action — requires legal/compliance review",
    "severe_frustration": "High frustration signal — risk of churn/reputation damage",
    "safety_concern": "Safety concern detected — immediate human attention required",
    "data_breach": "Data/security concern — requires security team review",
    "always_escalate_intent": "Intent '{intent}' always requires human review (billing/suspension)",
    "low_confidence": "Classifier confidence below threshold — uncertain intent",
}


class EscalationRouter:
    """
    Hybrid escalation router:
    1. Hard rules (billing, suspension, legal → always escalate)
    2. Keyword triggers (anger, legal threats, safety)
    3. Confidence threshold (low confidence → escalate)
    """

    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or settings.escalation_confidence_threshold

    def route(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
    ) -> dict:
        """
        Determine escalation decision.

        Returns:
            {
                should_escalate: bool,
                reason: str,
                trigger_type: str,
                priority: "low" | "medium" | "high" | "urgent"
            }
        """
        text_lower = customer_message.lower()

        # 1. Check always-escalate intents
        if intent in ALWAYS_ESCALATE:
            return {
                "should_escalate": True,
                "reason": ESCALATION_REASONS["always_escalate_intent"].format(intent=intent),
                "trigger_type": "always_escalate_intent",
                "priority": "high",
            }

        # 2. Check hard keyword triggers
        for trigger_type, keywords in ESCALATION_TRIGGERS.items():
            if any(kw in text_lower for kw in keywords):
                priority = "urgent" if trigger_type == "safety_concern" else "high"
                return {
                    "should_escalate": True,
                    "reason": ESCALATION_REASONS[trigger_type],
                    "trigger_type": trigger_type,
                    "priority": priority,
                }

        # 3. Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                "should_escalate": True,
                "reason": ESCALATION_REASONS["low_confidence"],
                "trigger_type": "low_confidence",
                "priority": "medium",
            }

        # 4. Auto-handle
        return {
            "should_escalate": False,
            "reason": f"Routine {intent} — safe to auto-handle with confidence {confidence:.2f}",
            "trigger_type": "none",
            "priority": "low",
        }


# Global instance
router = EscalationRouter()
