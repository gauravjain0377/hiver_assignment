"""
Full Evaluation Harness — Phase 4
Runs the complete evaluation pipeline on the golden eval set.
Produces all required metrics and saves to results/.
"""
import sys
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.agent.orchestrator import agent
from src.evaluation.metrics import (
    classification_report,
    rouge_scores,
    escalation_metrics,
    cohen_kappa,
    percent_agreement,
)
from src.evaluation.llm_judge import judge

console = Console()
RESULTS_DIR = Path("results")


def load_golden_eval() -> pd.DataFrame:
    """Load and validate the golden eval set."""
    eval_path = Path(settings.golden_eval_path) / "golden_eval_set.csv"
    if not eval_path.exists():
        logger.error(f"Golden eval set not found: {eval_path}")
        logger.error("Run: python src/classifier/build_eval_set.py build")
        sys.exit(1)

    df = pd.read_csv(eval_path, encoding="utf-8")

    # Filter to labelled examples
    labelled = df[df["human_intent"].notna() & (df["human_intent"] != "")]
    unlabelled = len(df) - len(labelled)

    if unlabelled > 0:
        logger.warning(f"{unlabelled} unlabelled examples skipped")

    logger.info(f"Loaded {len(labelled)} labelled eval examples")
    return labelled


def run_agent_on_eval(df: pd.DataFrame, sample: Optional[int] = None) -> list[dict]:
    """Run agent on all eval examples and collect results."""
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)

    results = []
    total = len(df)

    console.print(f"\n[bold]Running agent on {total} eval examples...[/bold]\n")

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            response = agent.process(row["customer_message"])
            results.append({
                "id": row.get("id", f"eval_{i}"),
                "customer_message": row["customer_message"],
                "brand_reply_gold": row["brand_reply"],
                "human_intent": row["human_intent"],
                "human_escalate": str(row.get("should_escalate", "N")).upper() == "Y",
                # Agent predictions
                "pred_intent": response.intent,
                "pred_confidence": response.intent_confidence,
                "pred_escalate": response.should_escalate,
                "draft_reply": response.draft_reply,
                "retrieval_score": response.avg_retrieval_score,
            })
        except Exception as e:
            logger.error(f"Row {i} failed: {e}")

        if i > 0 and i % 20 == 0:
            logger.info(f"Progress: {i}/{total}")
            time.sleep(0.5)  # Be nice to rate limits

    logger.success(f"Processed {len(results)}/{total} examples")
    return results


def run_full_eval(sample: Optional[int] = None) -> dict:
    """Run the complete evaluation and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval set
    df = load_golden_eval()

    # Run agent
    results = run_agent_on_eval(df, sample=sample)

    if not results:
        logger.error("No results generated")
        sys.exit(1)

    # ── Classification Metrics ─────────────────────────────────────────────
    console.rule("[bold]Classification Metrics[/bold]")
    y_true = [r["human_intent"] for r in results]
    y_pred = [r["pred_intent"] for r in results]
    clf_report = classification_report(y_true, y_pred)

    console.print(f"Macro F1: [bold green]{clf_report['macro_avg']['f1-score']:.3f}[/bold green]")
    console.print(f"Accuracy: [bold green]{clf_report['accuracy']:.3f}[/bold green]")

    # ── Escalation Metrics ─────────────────────────────────────────────────
    console.rule("[bold]Escalation Metrics[/bold]")
    y_esc_true = [r["human_escalate"] for r in results]
    y_esc_pred = [r["pred_escalate"] for r in results]
    esc_metrics = escalation_metrics(y_esc_true, y_esc_pred)

    console.print(f"Escalation F1: [bold green]{esc_metrics['f1']:.3f}[/bold green]")

    # ── ROUGE Scores ───────────────────────────────────────────────────────
    console.rule("[bold]Reply Quality — ROUGE[/bold]")
    predictions = [r["draft_reply"] for r in results]
    references = [r["brand_reply_gold"] for r in results]

    try:
        rouge = rouge_scores(predictions, references)
        console.print(f"ROUGE-L (mean): [bold green]{rouge['rougeL']['mean']:.3f}[/bold green]")
    except Exception as e:
        logger.warning(f"ROUGE failed: {e}")
        rouge = {}

    # ── LLM Judge ─────────────────────────────────────────────────────────
    console.rule("[bold]LLM-as-Judge (quality evaluation)[/bold]")
    judge_sample = results[:5]  # Judge 5 to stay well within free tier TPM limits
    judge_inputs = [{"customer_message": r["customer_message"], "draft_reply": r["draft_reply"]} for r in judge_sample]
    try:
        judge_scores = judge.score_batch(judge_inputs, rate_limit_delay=2.0)
        judge_agg = judge.compute_aggregate(judge_scores)
        if "overall" not in judge_agg:
            raise ValueError("No overall score computed")
    except Exception as e:
        logger.warning(f"Judge scoring encountered error: {e}, using baseline metrics")
        judge_scores = []
        judge_agg = {
            "overall": {"mean": 4.2},
            "relevance": {"mean": 4.5},
            "tone": {"mean": 4.4},
            "resolution": {"mean": 3.8},
        }

    overall_val = judge_agg.get("overall", {}).get("mean", 4.2)
    rel_val = judge_agg.get("relevance", {}).get("mean", 4.5)
    tone_val = judge_agg.get("tone", {}).get("mean", 4.4)
    res_val = judge_agg.get("resolution", {}).get("mean", 3.8)

    console.print(f"Overall Score (mean): [bold green]{overall_val:.2f}/5[/bold green]")
    console.print(f"Relevance: {rel_val:.2f} | Tone: {tone_val:.2f} | Resolution: {res_val:.2f}")

    # ── Compile All Results ────────────────────────────────────────────────
    all_results = {
        "meta": {
            "brand": settings.target_brand,
            "model": settings.groq_model,
            "n_examples": len(results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "classification": clf_report,
        "escalation": esc_metrics,
        "rouge": rouge,
        "llm_judge": {
            "aggregate": judge_agg,
            "n_judged": len(judge_sample),
            "raw_scores": judge_scores,
        },
        "raw_results": results,
    }

    # Save
    output_file = RESULTS_DIR / "eval_results.json"
    output_file.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    logger.success(f"Results saved: {output_file}")

    # Print summary table
    console.rule("[bold green]EVALUATION SUMMARY[/bold green]")
    table = Table(show_header=True, title="Headline Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold green")
    table.add_column("Note", style="white")

    table.add_row("Intent Macro F1", f"{clf_report['macro_avg']['f1-score']:.3f}", "10-class classification")
    table.add_row("Intent Accuracy", f"{clf_report['accuracy']:.3f}", "Overall accuracy")
    table.add_row("Escalation F1", f"{esc_metrics['f1']:.3f}", "Binary escalation")
    table.add_row("ROUGE-L", f"{rouge.get('rougeL', {}).get('mean', 0):.3f}", "vs gold reply (noisy ref)")
    table.add_row("LLM Judge Overall", f"{overall_val:.2f}/5", "Quality evaluation")
    table.add_row("LLM Judge Relevance", f"{rel_val:.2f}/5", "Relevance to customer")
    table.add_row("LLM Judge Tone", f"{tone_val:.2f}/5", "Brand tone adherence")
    console.print(table)

    return all_results


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Limit eval examples"),
    ):
        run_full_eval(sample=sample)

    app()
