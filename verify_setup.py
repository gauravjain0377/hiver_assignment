"""
Setup Verification Script
Run this after setting up .env to verify all connections work.
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def check_env():
    """Check all required environment variables."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.config import settings
        return settings, None
    except Exception as e:
        return None, str(e)


def check_groq(settings):
    """Test Groq API connection."""
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=5,
        )
        return True, response.choices[0].message.content.strip()
    except Exception as e:
        return False, str(e)


def check_kaggle(settings):
    """Test Kaggle credentials."""
    try:
        import os
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return True, f"Authenticated as {settings.kaggle_username}"
    except Exception as e:
        return False, str(e)


def check_qdrant(settings):
    """Test Qdrant connection."""
    if not settings.qdrant_url:
        return None, "QDRANT_URL not set (optional for local dev)"
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        info = client.get_collections()
        return True, f"Connected. Collections: {len(info.collections)}"
    except Exception as e:
        return False, str(e)


def check_embeddings():
    """Test local sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(["test"])
        return True, f"Embedding dim: {emb.shape[1]}"
    except Exception as e:
        return False, str(e)


def main():
    console.rule("[bold blue]Hiver AI Support Agent — Setup Verification[/bold blue]")

    # 1. Check environment
    console.print("\n[bold]Checking environment variables...[/bold]")
    settings, err = check_env()
    if err:
        console.print(f"[red]❌ Config error: {err}[/red]")
        console.print("[yellow]Make sure you copied .env.example to .env and filled in the values[/yellow]")
        sys.exit(1)
    console.print("[green]✓ Environment variables loaded[/green]")

    # Build results table
    table = Table(title="\nService Check Results", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="white")

    checks = [
        ("Groq LLM API", lambda: check_groq(settings)),
        ("Kaggle API", lambda: check_kaggle(settings)),
        ("Qdrant Vector DB", lambda: check_qdrant(settings)),
        ("Local Embeddings", lambda: check_embeddings()),
    ]

    all_pass = True
    for service, check_fn in checks:
        try:
            result, detail = check_fn()
            if result is True:
                table.add_row(service, "[green]✓ PASS[/green]", str(detail))
            elif result is None:
                table.add_row(service, "[yellow]⚠ SKIP[/yellow]", str(detail))
            else:
                table.add_row(service, "[red]❌ FAIL[/red]", str(detail))
                all_pass = False
        except Exception as e:
            table.add_row(service, "[red]❌ ERROR[/red]", str(e))
            all_pass = False

    console.print(table)

    if all_pass:
        console.print("\n[bold green]✅ All checks passed! Ready to start Phase 1.[/bold green]")
        console.print("[cyan]Next: python src/data/download.py[/cyan]\n")
    else:
        console.print("\n[bold red]⚠ Some checks failed. Fix the issues above before proceeding.[/bold red]")
        console.print("See .env.example for setup instructions.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
