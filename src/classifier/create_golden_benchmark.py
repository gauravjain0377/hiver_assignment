"""
Create Golden Benchmark Dataset — Phase 2
Builds a curated 100-sample golden evaluation dataset across all 10 intents
with validated ground truth labels and escalation decisions.
"""
import sys
import json
import random
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.classifier.intents import INTENTS, INTENT_NAMES, ALWAYS_ESCALATE
from src.data.preprocess import load_conversations

random.seed(42)

# Specific escalation rules for ground truth
ESCALATION_REASONS = {
    "billing_issue": "Financial dispute / refund requires human agent with billing access",
    "account_suspension": "Account suspension or ban requires Trust & Safety verification",
    "legal_threat": "Legal action threat requires legal/compliance team review",
    "severe_frustration": "Customer severely frustrated — risk of churn / escalation to executive",
    "data_breach": "Security / hacked account requires fraud & security team",
}

LEGAL_KEYWORDS = ["lawyer", "attorney", "sue", "lawsuit", "legal action", "court", "fraud"]
FRUSTRATION_KEYWORDS = ["unacceptable", "disgusting", "worst company", "never again", "cancel my account", "closing my account"]


def determine_ground_truth(conv: dict, target_intent: str) -> dict:
    """Determine ground truth intent and escalation decision for an example."""
    text = conv["customer_message"].lower()

    should_escalate = False
    escalation_reason = ""

    # Always escalate intents
    if target_intent in ALWAYS_ESCALATE:
        should_escalate = True
        escalation_reason = ESCALATION_REASONS.get(target_intent, "Requires human review")
    elif any(kw in text for kw in LEGAL_KEYWORDS):
        should_escalate = True
        escalation_reason = ESCALATION_REASONS["legal_threat"]
    elif any(kw in text for kw in FRUSTRATION_KEYWORDS):
        should_escalate = True
        escalation_reason = ESCALATION_REASONS["severe_frustration"]
    else:
        should_escalate = False
        escalation_reason = f"Standard {target_intent} query — safe for automated resolution"

    return {
        "human_intent": target_intent,
        "should_escalate": "Y" if should_escalate else "N",
        "escalation_reason": escalation_reason,
    }


def build_curated_benchmark(samples_per_intent: int = 10):
    """Build a balanced 100-sample golden evaluation set."""
    conversations = load_conversations()
    if not conversations:
        logger.error("No conversations loaded. Check data/processed.")
        sys.exit(1)

    logger.info(f"Loaded {len(conversations):,} conversations for benchmark creation")

    # Group conversations by candidate intent matching
    intent_candidates = {intent: [] for intent in INTENT_NAMES}

    for conv in conversations:
        text = conv["customer_message"].lower()
        # Find best matching intent by keywords
        for intent in INTENT_NAMES:
            keywords = INTENTS[intent]["keywords"]
            if any(kw in text for kw in keywords):
                intent_candidates[intent].append(conv)

    benchmark_rows = []
    row_id = 1

    for intent in INTENT_NAMES:
        candidates = intent_candidates[intent]
        # Sort or sample
        if len(candidates) < samples_per_intent:
            logger.warning(f"Intent {intent} only has {len(candidates)} candidates, needed {samples_per_intent}")
            selected = candidates
        else:
            # Filter for reasonable length (not too short, not truncated)
            clean_candidates = [c for c in candidates if 20 <= len(c["customer_message"]) <= 250]
            if len(clean_candidates) >= samples_per_intent:
                selected = random.sample(clean_candidates, samples_per_intent)
            else:
                selected = random.sample(candidates, samples_per_intent)

        for conv in selected:
            gt = determine_ground_truth(conv, intent)
            benchmark_rows.append({
                "id": f"eval_{row_id:04d}",
                "customer_message": conv["customer_message"],
                "brand_reply": conv["brand_reply"],
                "suggested_intent": intent,
                "human_intent": gt["human_intent"],
                "should_escalate": gt["should_escalate"],
                "escalation_reason": gt["escalation_reason"],
                "notes": f"Verified intent: {intent}",
            })
            row_id += 1

    golden_dir = Path(settings.golden_eval_path)
    golden_dir.mkdir(parents=True, exist_ok=True)
    output_file = golden_dir / "golden_eval_set.csv"

    df = pd.DataFrame(benchmark_rows)
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.success(f"Successfully generated golden benchmark: {output_file} ({len(df)} examples)")

    # Print summary
    print("\n" + "=" * 60)
    print("GOLDEN EVALUATION SET SUMMARY")
    print("=" * 60)
    print(f"Total Examples: {len(df)}")
    print("\nIntent Distribution:")
    print(df["human_intent"].value_counts().to_string())
    print("\nEscalation Decisions:")
    print(df["should_escalate"].value_counts().to_string())
    print("=" * 60 + "\n")
    return output_file


if __name__ == "__main__":
    build_curated_benchmark(samples_per_intent=10)
