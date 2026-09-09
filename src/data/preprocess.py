"""
Data Preprocessing — Phase 1
Filters raw Twitter data for target brand, reconstructs conversation threads,
cleans text, and saves processed data.
"""
import sys
import re
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_tweet(text: str) -> str:
    """Remove URLs, mentions, extra whitespace from tweet text."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove @mentions (keep them as context clue but strip the @ symbol)
    text = re.sub(r"@\w+", "", text)
    # Remove special chars but keep punctuation
    text = re.sub(r"[^\w\s\.\,\!\?\-\'\"]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_brand_tweet(row: pd.Series, brand: str) -> bool:
    """Check if a tweet is FROM the brand (support response)."""
    author = str(row.get("author_id", "")).lower()
    return brand.lower() in author


def is_customer_tweet(row: pd.Series, brand: str) -> bool:
    """Check if a tweet is TO the brand (customer message)."""
    inbound = row.get("inbound", False)
    return bool(inbound)


# ─── Thread Reconstruction ────────────────────────────────────────────────────

def reconstruct_threads(df: pd.DataFrame, brand: str) -> list[dict]:
    """
    Reconstruct (customer_message, brand_reply) pairs from tweet threads.
    Returns a list of conversation dicts.
    """
    logger.info("Building tweet index...")
    tweet_index = df.set_index("tweet_id").to_dict("index")

    conversations = []

    # Find all brand replies
    brand_tweets = df[df["author_id"].str.lower().str.contains(brand.lower(), na=False)]
    logger.info(f"Found {len(brand_tweets):,} tweets from {brand}")

    for _, brand_tweet in tqdm(brand_tweets.iterrows(), total=len(brand_tweets), desc="Reconstructing threads"):
        in_response_to = brand_tweet.get("in_response_to_tweet_id")
        if pd.isna(in_response_to):
            continue

        # Find the customer message this brand tweet replied to
        in_response_to = int(in_response_to)
        if in_response_to not in tweet_index:
            continue

        customer_tweet = tweet_index[in_response_to]

        customer_text = clean_tweet(customer_tweet.get("text", ""))
        brand_text = clean_tweet(brand_tweet.get("text", ""))

        if not customer_text or not brand_text:
            continue
        if len(customer_text) < 10 or len(brand_text) < 10:
            continue

        conversations.append({
            "conversation_id": str(brand_tweet.get("tweet_id", "")),
            "customer_tweet_id": str(in_response_to),
            "brand_tweet_id": str(brand_tweet.get("tweet_id", "")),
            "customer_message": customer_text,
            "brand_reply": brand_text,
            "created_at": str(brand_tweet.get("created_at", "")),
            "brand": brand,
        })

    logger.success(f"Reconstructed {len(conversations):,} conversation pairs")
    return conversations


# ─── Main Processing Pipeline ─────────────────────────────────────────────────

def process_data(
    brand: Optional[str] = None,
    sample_size: Optional[int] = None,
    chunk_size: int = 100_000,
) -> Path:
    """
    Full preprocessing pipeline:
    1. Load raw CSV in chunks (handles large file)
    2. Filter for target brand
    3. Reconstruct conversation threads
    4. Save to processed directory
    """
    brand = brand or settings.target_brand
    raw_path = Path(settings.data_raw_path) / "twcs.csv"
    processed_path = Path(settings.data_processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_path / f"{brand.lower()}_conversations.json"

    if output_file.exists():
        logger.info(f"Processed data already exists: {output_file}")
        existing = json.loads(output_file.read_text())
        logger.info(f"Loaded {len(existing):,} conversations")
        return output_file

    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        logger.error("Run: python src/data/download.py first")
        sys.exit(1)

    logger.info(f"Processing data for brand: {brand}")
    logger.info(f"Reading {raw_path} in chunks of {chunk_size:,}...")

    chunks = []
    total_rows = 0

    for chunk in tqdm(
        pd.read_csv(raw_path, chunksize=chunk_size, dtype=str, low_memory=False),
        desc="Reading chunks"
    ):
        total_rows += len(chunk)
        # Keep only rows relevant to this brand
        mask = (
            chunk["author_id"].str.lower().str.contains(brand.lower(), na=False) |
            chunk["in_response_to_tweet_id"].notna()
        )
        chunks.append(chunk[mask])

    logger.info(f"Total rows in dataset: {total_rows:,}")
    df = pd.concat(chunks, ignore_index=True)

    # Convert types
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    df["inbound"] = df["inbound"].map({"True": True, "False": False, True: True, False: False})

    logger.info(f"Filtered to {len(df):,} relevant rows for {brand}")

    # Reconstruct threads
    conversations = reconstruct_threads(df, brand)

    if sample_size and len(conversations) > sample_size:
        import random
        random.seed(42)
        conversations = random.sample(conversations, sample_size)
        logger.info(f"Sampled down to {sample_size:,} conversations")

    # Save
    output_file.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
    logger.success(f"Saved {len(conversations):,} conversations to {output_file}")

    # Save stats
    stats = {
        "brand": brand,
        "total_raw_rows": total_rows,
        "total_conversations": len(conversations),
        "avg_customer_message_length": sum(len(c["customer_message"]) for c in conversations) / max(len(conversations), 1),
        "avg_brand_reply_length": sum(len(c["brand_reply"]) for c in conversations) / max(len(conversations), 1),
    }
    stats_file = processed_path / f"{brand.lower()}_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats: {stats}")

    return output_file


def load_conversations(brand: Optional[str] = None) -> list[dict]:
    """Load processed conversations for a brand."""
    brand = brand or settings.target_brand
    processed_path = Path(settings.data_processed_path)
    output_file = processed_path / f"{brand.lower()}_conversations.json"

    if not output_file.exists():
        logger.error(f"Processed data not found: {output_file}")
        logger.error("Run: python src/data/preprocess.py first")
        return []

    conversations = json.loads(output_file.read_text())
    logger.info(f"Loaded {len(conversations):,} conversations for {brand}")
    return conversations


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        brand: str = typer.Option(settings.target_brand, "--brand", "-b"),
        sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample N conversations"),
    ):
        process_data(brand=brand, sample_size=sample)

    app()
