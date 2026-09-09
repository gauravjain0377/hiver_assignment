"""
Agent Orchestrator — Phase 3
Main pipeline: classify → retrieve → draft → route
The single entry point for processing any customer message.
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.classifier.classifier import classifier
from src.rag.vector_store import vector_store
from src.agent.drafter import drafter
from src.agent.router import router


@dataclass
class AgentResponse:
    """Full agent response for a customer message."""
    # Input
    customer_message: str

    # Classification
    intent: str
    intent_label: str
    intent_confidence: float
    classification_method: str  # "keyword" | "llm"

    # Reply
    draft_reply: str
    retrieved_examples_count: int
    avg_retrieval_score: float
    fallback_used: bool

    # Routing
    should_escalate: bool
    escalation_reason: str
    escalation_trigger: str
    priority: str  # "low" | "medium" | "high" | "urgent"

    # Metadata
    processing_time_ms: float
    brand: str
    retrieved_examples: list = None

    def to_dict(self) -> dict:
        return asdict(self)


class AgentOrchestrator:
    """
    Main agent pipeline.
    Combines classifier, RAG retriever, drafter, and router.
    """

    def process(self, customer_message: str) -> AgentResponse:
        """
        Process a single customer message end-to-end.

        Pipeline:
        1. Classify intent
        2. Retrieve similar historical cases (for RAG)
        3. Draft reply (grounded in retrieved cases)
        4. Route (auto-handle vs escalate)
        5. Return full AgentResponse
        """
        start = time.time()

        if not customer_message or not customer_message.strip():
            raise ValueError("customer_message cannot be empty")

        # Step 1: Classify intent
        classification = classifier.predict(customer_message)
        intent = classification["intent"]
        confidence = classification["confidence"]
        method = classification["method"]

        logger.debug(f"Intent: {intent} ({confidence:.2f}) via {method}")

        # Step 2: Retrieve similar cases
        retrieved = vector_store.search(customer_message, top_k=settings.rag_top_k)

        # Step 3: Draft reply
        draft_result = drafter.draft(
            customer_message=customer_message,
            intent=intent,
            retrieved_examples=retrieved,
        )

        # Step 4: Route escalation
        routing = router.route(
            customer_message=customer_message,
            intent=intent,
            confidence=confidence,
        )

        elapsed_ms = (time.time() - start) * 1000

        return AgentResponse(
            customer_message=customer_message,
            intent=intent,
            intent_label=classification["intent_label"],
            intent_confidence=confidence,
            classification_method=method,
            draft_reply=draft_result["draft"],
            retrieved_examples_count=draft_result["retrieved_count"],
            avg_retrieval_score=draft_result["avg_retrieval_score"],
            fallback_used=draft_result["fallback_used"],
            retrieved_examples=retrieved,
            should_escalate=routing["should_escalate"],
            escalation_reason=routing["reason"],
            escalation_trigger=routing["trigger_type"],
            priority=routing["priority"],
            processing_time_ms=round(elapsed_ms, 1),
            brand=settings.target_brand,
        )

    def process_batch(self, messages: list[str]) -> list[AgentResponse]:
        """Process multiple messages."""
        results = []
        for msg in messages:
            try:
                results.append(self.process(msg))
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
        return results


# Global instance
agent = AgentOrchestrator()


if __name__ == "__main__":
    import json
    import typer

    app = typer.Typer()

    @app.command()
    def demo(message: str = typer.Argument("I can't log into my Apple account and I need help urgently")):
        """Run a demo of the full agent pipeline."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        console.rule("[bold blue]Hiver AI Support Agent Demo[/bold blue]")

        console.print(f"\n[bold]Customer message:[/bold] {message}\n")
        console.print("Processing...\n")

        response = agent.process(message)

        table = Table(show_header=True, title="Agent Response")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Intent", f"{response.intent_label} ({response.intent})")
        table.add_row("Confidence", f"{response.intent_confidence:.0%} ({response.classification_method})")
        table.add_row("Escalate?", f"{'[red]YES[/red]' if response.should_escalate else '[green]NO[/green]'}")
        table.add_row("Escalation Reason", response.escalation_reason)
        table.add_row("Priority", response.priority.upper())
        table.add_row("Retrieved Examples", str(response.retrieved_examples_count))
        table.add_row("Avg Retrieval Score", f"{response.avg_retrieval_score:.3f}")
        table.add_row("Processing Time", f"{response.processing_time_ms:.0f}ms")

        console.print(table)
        console.print(
            Panel(
                response.draft_reply,
                title="[bold green]Draft Reply[/bold green]",
                border_style="green",
            )
        )

    app()
