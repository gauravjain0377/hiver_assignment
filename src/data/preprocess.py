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
    chunk_size: int = 200_000,
) -> Path:
    """
    Two-pass efficient preprocessing pipeline:
    1. Pass 1: Collect brand replies and set of parent customer tweet IDs.
    2. Pass 2: Collect parent customer tweets by ID.
    3. Reconstruct (customer_message, brand_reply) pairs and save.
    """
    brand = brand or settings.target_brand
    raw_path = Path(settings.data_raw_path) / "twcs.csv"
    processed_path = Path(settings.data_processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_path / f"{brand.lower()}_conversations.json"

    if output_file.exists():
        logger.info(f"Processed data already exists: {output_file}")
        existing = json.loads(output_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(existing):,} conversations")
        return output_file

    if not raw_path.exists():
        logger.error(f"Raw data not found: {raw_path}")
        logger.error("Run: python src/data/download.py first")
        sys.exit(1)

    logger.info(f"Processing data for brand: {brand}")
    logger.info("Pass 1: Identifying brand replies and target customer tweet IDs...")

    brand_replies = []
    parent_ids = set()
    total_rows = 0

    for chunk in tqdm(
        pd.read_csv(raw_path, chunksize=chunk_size, dtype=str, low_memory=False),
        desc="Pass 1 (Brand replies)"
    ):
        total_rows += len(chunk)
        # Filter for tweets authored by brand that are replies
        mask = (
            chunk["author_id"].str.lower().str.contains(brand.lower(), na=False) &
            chunk["in_response_to_tweet_id"].notna()
        )
        brand_chunk = chunk[mask]
        if not brand_chunk.empty:
            brand_replies.append(brand_chunk)
            # Parse parent IDs
            clean_parent_ids = pd.to_numeric(brand_chunk["in_response_to_tweet_id"], errors="coerce").dropna().astype(int)
            parent_ids.update(clean_parent_ids.tolist())

    logger.info(f"Total raw tweets scanned: {total_rows:,}")
    if not brand_replies:
        logger.warning(f"No replies found for brand: {brand}")
        return output_file

    df_brand = pd.concat(brand_replies, ignore_index=True)
    df_brand["tweet_id"] = pd.to_numeric(df_brand["tweet_id"], errors="coerce")
    df_brand["in_response_to_tweet_id"] = pd.to_numeric(df_brand["in_response_to_tweet_id"], errors="coerce")
    logger.info(f"Found {len(df_brand):,} replies from {brand} addressing {len(parent_ids):,} customer tweets")

    # Pass 2: Extract customer tweets matching parent_ids
    logger.info("Pass 2: Extracting matched customer tweets...")
    customer_tweets = {}

    for chunk in tqdm(
        pd.read_csv(raw_path, chunksize=chunk_size, dtype=str, low_memory=False),
        desc="Pass 2 (Customer tweets)"
    ):
        chunk["tweet_id_num"] = pd.to_numeric(chunk["tweet_id"], errors="coerce")
        matched = chunk[chunk["tweet_id_num"].isin(parent_ids)]
        for _, row in matched.iterrows():
            tid = int(row["tweet_id_num"])
            customer_tweets[tid] = row["text"]

    logger.info(f"Retrieved text for {len(customer_tweets):,} customer tweets")

    # Reconstruct pairs
    conversations = []
    for _, b_row in tqdm(df_brand.iterrows(), total=len(df_brand), desc="Pairing conversations"):
        in_resp = b_row["in_response_to_tweet_id"]
        if pd.isna(in_resp):
            continue
        in_resp_id = int(in_resp)
        raw_cust_text = customer_tweets.get(in_resp_id)
        if not raw_cust_text:
            continue

        c_text = clean_tweet(raw_cust_text)
        b_text = clean_tweet(b_row.get("text", ""))

        if len(c_text) < 10 or len(b_text) < 10:
            continue

        conversations.append({
            "conversation_id": str(int(b_row["tweet_id"])) if pd.notna(b_row["tweet_id"]) else str(b_row["tweet_id"]),
            "customer_tweet_id": str(in_resp_id),
            "brand_tweet_id": str(int(b_row["tweet_id"])) if pd.notna(b_row["tweet_id"]) else str(b_row["tweet_id"]),
            "customer_message": c_text,
            "brand_reply": b_text,
            "created_at": str(b_row.get("created_at", "")),
            "brand": brand,
        })

    logger.success(f"Reconstructed {len(conversations):,} clean conversation pairs")

    if sample_size and len(conversations) > sample_size:
        import random
        random.seed(42)
        conversations = random.sample(conversations, sample_size)
        logger.info(f"Sampled down to {sample_size:,} conversations")

    # Save
    output_file.write_text(json.dumps(conversations, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Saved {len(conversations):,} conversations to {output_file}")

    stats = {
        "brand": brand,
        "total_raw_rows": total_rows,
        "total_conversations": len(conversations),
        "avg_customer_message_length": sum(len(c["customer_message"]) for c in conversations) / max(len(conversations), 1),
        "avg_brand_reply_length": sum(len(c["brand_reply"]) for c in conversations) / max(len(conversations), 1),
    }
    stats_file = processed_path / f"{brand.lower()}_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
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

    conversations = json.loads(output_file.read_text(encoding="utf-8"))
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
