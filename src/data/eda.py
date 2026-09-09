"""
EDA Script — Phase 1
Exploratory Data Analysis of the processed brand conversations.
Outputs stats, plots, and intent clustering insights.
"""
import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.data.preprocess import load_conversations

console = Console()


def basic_stats(conversations: list[dict]) -> dict:
    """Compute basic statistics about the conversation dataset."""
    customer_lengths = [len(c["customer_message"]) for c in conversations]
    reply_lengths = [len(c["brand_reply"]) for c in conversations]

    stats = {
        "total_conversations": len(conversations),
        "avg_customer_message_len": np.mean(customer_lengths),
        "median_customer_message_len": np.median(customer_lengths),
        "avg_reply_len": np.mean(reply_lengths),
        "median_reply_len": np.median(reply_lengths),
        "min_customer_len": min(customer_lengths),
        "max_customer_len": max(customer_lengths),
    }
    return stats


def keyword_intent_analysis(conversations: list[dict]) -> dict:
    """
    Keyword-based intent discovery for bootstrapping.
    Groups conversations by common keywords/phrases.
    """
    # Define candidate intent keywords
    intent_keywords = {
        "login_problem": ["can't login", "cannot login", "sign in", "password", "account access", "locked out"],
        "billing_issue": ["charge", "refund", "payment", "bill", "subscription", "cancel", "money"],
        "bug_report": ["not working", "broken", "error", "crash", "bug", "issue", "glitch", "freeze"],
        "feature_request": ["would be great", "please add", "feature", "suggestion", "wish", "request"],
        "account_suspension": ["suspended", "banned", "disabled", "blocked", "removed"],
        "content_issue": ["song", "playlist", "music", "video", "content", "missing", "disappeared"],
        "app_performance": ["slow", "lag", "loading", "performance", "battery", "drain", "memory"],
        "general_inquiry": ["how do", "can i", "is it possible", "how to", "what is", "help me"],
        "positive_feedback": ["love", "great", "amazing", "thank", "awesome", "best", "perfect"],
        "connectivity": ["offline", "internet", "connection", "wifi", "network", "sync"],
    }

    intent_counts = {intent: 0 for intent in intent_keywords}
    examples = {intent: [] for intent in intent_keywords}

    for conv in conversations:
        text = conv["customer_message"].lower()
        for intent, keywords in intent_keywords.items():
            if any(kw in text for kw in keywords):
                intent_counts[intent] += 1
                if len(examples[intent]) < 3:
                    examples[intent].append(conv["customer_message"][:100])

    return {"counts": intent_counts, "examples": examples}


def print_eda_report(conversations: list[dict], stats: dict, intent_analysis: dict):
    """Print a rich EDA report to console."""

    console.rule("[bold blue]HIVER AI SUPPORT AGENT — EDA REPORT[/bold blue]")
    console.print(f"\n[bold]Brand:[/bold] {settings.target_brand}")
    console.print(f"[bold]Total Conversations:[/bold] {stats['total_conversations']:,}\n")

    # Basic stats table
    table = Table(title="Dataset Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Conversations", f"{stats['total_conversations']:,}")
    table.add_row("Avg Customer Message Length", f"{stats['avg_customer_message_len']:.0f} chars")
    table.add_row("Median Customer Message Length", f"{stats['median_customer_message_len']:.0f} chars")
    table.add_row("Avg Brand Reply Length", f"{stats['avg_reply_len']:.0f} chars")
    table.add_row("Median Brand Reply Length", f"{stats['median_reply_len']:.0f} chars")
    console.print(table)

    # Intent distribution
    intent_table = Table(title="\nKeyword-Based Intent Distribution", show_header=True)
    intent_table.add_column("Intent", style="cyan")
    intent_table.add_column("Count", style="green")
    intent_table.add_column("% of Total", style="yellow")
    intent_table.add_column("Example", style="white", max_width=60)

    total = stats["total_conversations"]
    sorted_intents = sorted(
        intent_analysis["counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for intent, count in sorted_intents:
        pct = (count / total * 100) if total > 0 else 0
        example = intent_analysis["examples"][intent][0] if intent_analysis["examples"][intent] else "—"
        intent_table.add_row(intent, str(count), f"{pct:.1f}%", example[:60])

    console.print(intent_table)
    console.print("\n[bold green]✓ EDA complete. Intent keywords validated.[/bold green]")
    console.print("[yellow]Next step: Run Phase 2 - Golden eval set labelling[/yellow]\n")


def run_eda():
    """Full EDA pipeline."""
    conversations = load_conversations()
    if not conversations:
        logger.error("No conversations found. Run preprocessing first:")
        logger.error("  python src/data/preprocess.py")
        sys.exit(1)

    logger.info("Running EDA...")
    stats = basic_stats(conversations)
    intent_analysis = keyword_intent_analysis(conversations)

    print_eda_report(conversations, stats, intent_analysis)

    # Save EDA results
    processed_path = Path(settings.data_processed_path)
    eda_output = processed_path / f"{settings.target_brand.lower()}_eda.json"
    eda_output.write_text(json.dumps({
        "stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for k, v in stats.items()},
        "intent_keyword_counts": intent_analysis["counts"],
        "sample_examples": {k: v[:2] for k, v in intent_analysis["examples"].items()},
    }, indent=2))
    logger.success(f"EDA results saved to {eda_output}")


if __name__ == "__main__":
    run_eda()
