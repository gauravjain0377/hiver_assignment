"""
Data Download Script — Phase 1
Downloads the Twitter Customer Support dataset from Kaggle.
"""
import os
import sys
import zipfile
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


def setup_kaggle_credentials():
    """Set Kaggle credentials from environment variables."""
    if not settings.kaggle_username or not settings.kaggle_key:
        logger.error(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY in .env file.\n"
            "See .env.example for setup instructions."
        )
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    os.environ["KAGGLE_KEY"] = settings.kaggle_key
    logger.info(f"Kaggle credentials set for user: {settings.kaggle_username}")


def download_dataset():
    """Download the Twitter Customer Support dataset from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_path = Path(settings.data_raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    dataset_file = raw_path / "twcs.csv"
    if dataset_file.exists():
        logger.info(f"Dataset already exists at {dataset_file}. Skipping download.")
        return dataset_file

    logger.info("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset: thoughtvector/customer-support-on-twitter")
    logger.info("This is ~500MB — may take a few minutes...")

    api.dataset_download_files(
        "thoughtvector/customer-support-on-twitter",
        path=str(raw_path),
        unzip=True,
        quiet=False,
    )

    # Find the CSV file
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        # Check for zip and extract manually
        zip_files = list(raw_path.glob("*.zip"))
        if zip_files:
            logger.info("Extracting zip file...")
            with zipfile.ZipFile(zip_files[0], "r") as z:
                z.extractall(raw_path)
            csv_files = list(raw_path.glob("*.csv"))

    if not csv_files:
        logger.error("No CSV file found after download!")
        sys.exit(1)

    # Rename to standard name
    csv_file = csv_files[0]
    if csv_file.name != "twcs.csv":
        csv_file.rename(dataset_file)

    logger.success(f"Dataset downloaded successfully: {dataset_file}")
    logger.info(f"File size: {dataset_file.stat().st_size / (1024**2):.1f} MB")
    return dataset_file


def verify_download():
    """Quick verification that the download worked."""
    import pandas as pd

    raw_path = Path(settings.data_raw_path)
    csv_file = raw_path / "twcs.csv"

    if not csv_file.exists():
        logger.error("Dataset file not found! Run download first.")
        return False

    logger.info("Verifying dataset...")
    df = pd.read_csv(csv_file, nrows=5)
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Sample row:\n{df.iloc[0].to_dict()}")
    logger.success("Dataset verified successfully!")
    return True


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(verify: bool = typer.Option(False, "--verify", help="Only verify, don't download")):
        if verify:
            verify_download()
        else:
            setup_kaggle_credentials()
            download_dataset()
            verify_download()

    app()
