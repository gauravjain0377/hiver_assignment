"""
Golden Eval Set Builder — Phase 2
Samples and prepares conversations for hand-labelling.
Creates a CSV for easy labelling + a validation script.
"""
import sys
import json
import random
import csv
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.data.preprocess import load_conversations
from src.classifier.intents import INTENT_NAMES, INTENTS

console = Console()
random.seed(42)


def keyword_label(text: str) -> Optional[str]:
    """Quick keyword-based label for bootstrapping (not ground truth)."""
    text_lower = text.lower()
    for intent_name, intent_def in INTENTS.items():
        if any(kw in text_lower for kw in intent_def["keywords"]):
            return intent_name
    return "general_inquiry"


def stratified_sample(conversations: list[dict], total: int = 200) -> list[dict]:
    """
    Stratified sample to ensure all intents are represented.
    ~20 examples per intent = 200 total.
    """
    # Group by keyword-based label
    buckets = {intent: [] for intent in INTENT_NAMES}
    unlabelled = []

    for conv in conversations:
        label = keyword_label(conv["customer_message"])
        if label:
            buckets[label].append(conv)
        else:
            unlabelled.append(conv)

    # Sample proportionally
    per_intent = total // len(INTENT_NAMES)
    sampled = []

    for intent, convs in buckets.items():
        n = min(per_intent, len(convs))
        sampled.extend(random.sample(convs, n))
        logger.info(f"  {intent}: {n} samples (pool size: {len(convs)})")

    # Fill remainder from unlabelled
    remaining = total - len(sampled)
    if remaining > 0 and unlabelled:
        sampled.extend(random.sample(unlabelled, min(remaining, len(unlabelled))))

    random.shuffle(sampled)
    logger.success(f"Total sampled: {len(sampled)} conversations")
    return sampled


def build_golden_eval_set(total: int = 200):
    """
    Build the golden eval set CSV for hand-labelling.

    Output format:
    - id: unique identifier
    - customer_message: the raw customer tweet
    - brand_reply: the brand's actual historical reply
    - suggested_intent: keyword-based suggestion (you will correct this)
    - human_intent: BLANK — fill this in manually
    - should_escalate: BLANK — fill Y/N manually
    - escalation_reason: BLANK — fill if escalating
    - notes: BLANK — any notes
    """
    golden_path = Path(settings.golden_eval_path)
    golden_path.mkdir(parents=True, exist_ok=True)
    output_csv = golden_path / "golden_eval_set.csv"

    if output_csv.exists():
        logger.info(f"Golden eval set already exists: {output_csv}")
        existing = pd.read_csv(output_csv)
        logger.info(f"Rows: {len(existing)}")
        return output_csv

    conversations = load_conversations()
    if not conversations:
        logger.error("No conversations found. Run preprocessing first.")
        sys.exit(1)

    logger.info(f"Sampling {total} conversations for golden eval set...")
    sampled = stratified_sample(conversations, total=total)

    # Build CSV rows
    rows = []
    for i, conv in enumerate(sampled):
        rows.append({
            "id": f"eval_{i+1:04d}",
            "customer_message": conv["customer_message"],
            "brand_reply": conv["brand_reply"],
            "suggested_intent": keyword_label(conv["customer_message"]) or "general_inquiry",
            "human_intent": "",  # TO BE FILLED
            "should_escalate": "",  # Y or N
            "escalation_reason": "",  # Fill if Y
            "notes": "",
        })

    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.success(f"Golden eval set saved: {output_csv}")

    # Print labelling instructions
    console.rule("[bold yellow]LABELLING INSTRUCTIONS[/bold yellow]")
    console.print(
        "\n[bold]You need to manually label the 'human_intent' and 'should_escalate' columns.[/bold]\n"
        f"\nFile to edit: [cyan]{output_csv}[/cyan]\n"
    )

    # Print intent options
    table = Table(title="Valid Intent Values for 'human_intent' column")
    table.add_column("Intent Key", style="cyan")
    table.add_column("Label", style="green")
    table.add_column("Description", style="white")

    for intent_name, intent_def in INTENTS.items():
        table.add_row(intent_name, intent_def["label"], intent_def["description"])

    console.print(table)
    console.print(
        "\n[bold]For 'should_escalate':[/bold] Enter Y (yes, escalate to human) or N (no, auto-handle)\n"
        "[bold]For 'escalation_reason':[/bold] If Y, write a short reason (e.g., 'billing dispute', 'legal threat')\n"
        "\n[yellow]Tip:[/yellow] The 'suggested_intent' column is a keyword guess. Correct it where wrong.\n"
        "\n[bold green]After labelling, run:[/bold green] python src/classifier/validate_golden_eval.py\n"
    )

    return output_csv


def validate_golden_eval():
    """Validate that the golden eval set is properly labelled."""
    golden_path = Path(settings.golden_eval_path) / "golden_eval_set.csv"
    if not golden_path.exists():
        logger.error("Golden eval set not found. Run build first.")
        sys.exit(1)

    df = pd.read_csv(golden_path)
    valid_intents = set(INTENT_NAMES)

    errors = []
    warnings = []

    for i, row in df.iterrows():
        # Check human_intent
        if pd.isna(row.get("human_intent")) or row["human_intent"] == "":
            warnings.append(f"Row {i+1}: Missing human_intent")
        elif row["human_intent"] not in valid_intents:
            errors.append(f"Row {i+1}: Invalid intent '{row['human_intent']}'. Valid: {valid_intents}")

        # Check should_escalate
        if pd.isna(row.get("should_escalate")) or row["should_escalate"] == "":
            warnings.append(f"Row {i+1}: Missing should_escalate (Y/N)")
        elif str(row["should_escalate"]).upper() not in ("Y", "N"):
            errors.append(f"Row {i+1}: should_escalate must be Y or N, got '{row['should_escalate']}'")

    unlabelled = df[df["human_intent"] == ""].shape[0]
    labelled = len(df) - unlabelled

    console.rule("[bold]Golden Eval Set Validation[/bold]")
    console.print(f"Total rows: {len(df)}")
    console.print(f"Labelled: [green]{labelled}[/green]")
    console.print(f"Unlabelled: [yellow]{unlabelled}[/yellow]")

    if errors:
        console.print(f"\n[red]ERRORS ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  ❌ {e}")
    else:
        console.print("\n[green]✓ No validation errors[/green]")

    if warnings and unlabelled > 0:
        console.print(f"\n[yellow]Warnings (unlabelled rows): {len(warnings)}[/yellow]")

    if labelled >= 150:
        console.print(f"\n[bold green]✓ {labelled} labelled examples — ready for Phase 3![/bold green]")
    else:
        console.print(f"\n[bold red]Need at least 150 labelled examples. Currently: {labelled}[/bold red]")


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def build(total: int = typer.Option(200, "--total", "-n")):
        build_golden_eval_set(total=total)

    @app.command()
    def validate():
        validate_golden_eval()

    app()
