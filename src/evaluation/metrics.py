"""
Evaluation Metrics — Phase 4
Classification metrics, reply quality metrics, escalation accuracy.
"""
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.classifier.intents import INTENT_NAMES


# ─── Classification Metrics ───────────────────────────────────────────────────

def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """Compute per-intent F1, precision, recall + macro averages."""
    from sklearn.metrics import classification_report as sk_report, confusion_matrix

    labels = labels or INTENT_NAMES
    report = sk_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "per_intent": {
            intent: {
                "precision": report[intent]["precision"],
                "recall": report[intent]["recall"],
                "f1": report[intent]["f1-score"],
                "support": report[intent]["support"],
            }
            for intent in labels if intent in report
        },
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "accuracy": report["accuracy"],
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


# ─── Reply Quality Metrics ────────────────────────────────────────────────────

def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L scores between drafts and gold replies."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(score[k].fmeasure)

        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}
    except ImportError:
        logger.warning("rouge_score not installed, skipping ROUGE metrics")
        return {}


def bert_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore between drafts and gold replies."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return {
            "precision": {"mean": float(P.mean()), "std": float(P.std())},
            "recall": {"mean": float(R.mean()), "std": float(R.std())},
            "f1": {"mean": float(F1.mean()), "std": float(F1.std())},
        }
    except ImportError:
        logger.warning("bert_score not installed, skipping BERTScore metrics")
        return {}


# ─── Escalation Metrics ───────────────────────────────────────────────────────

def escalation_metrics(
    y_true_escalate: list[bool],
    y_pred_escalate: list[bool],
) -> dict:
    """Binary classification metrics for escalation decisions."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score
    )

    return {
        "accuracy": accuracy_score(y_true_escalate, y_pred_escalate),
        "precision": precision_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "recall": recall_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "f1": f1_score(y_true_escalate, y_pred_escalate, zero_division=0),
        "total": len(y_true_escalate),
        "escalated_true": sum(y_true_escalate),
        "escalated_pred": sum(y_pred_escalate),
    }


# ─── Human-Judge Agreement ────────────────────────────────────────────────────

def cohen_kappa(
    human_scores: list[int],
    llm_scores: list[int],
) -> float:
    """Compute Cohen's Kappa for human vs LLM judge agreement."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(human_scores, llm_scores))


def percent_agreement(
    human_scores: list[int],
    llm_scores: list[int],
    tolerance: int = 1,
) -> float:
    """Compute % agreement within tolerance."""
    agreements = sum(
        1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= tolerance
    )
    return agreements / len(human_scores) if human_scores else 0.0
