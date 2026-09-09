"""
Hiver AI Support Agent — Project Configuration
Loads and validates all environment variables using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")
    mistral_api_key: str = Field("", env="MISTRAL_API_KEY")

    # Vector DB
    qdrant_url: str = Field("", env="QDRANT_URL")
    qdrant_api_key: str = Field("", env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("hiver_support_tweets", env="QDRANT_COLLECTION_NAME")

    # Kaggle
    kaggle_username: str = Field("", env="KAGGLE_USERNAME")
    kaggle_key: str = Field("", env="KAGGLE_KEY")

    # HuggingFace
    hf_token: str = Field("", env="HF_TOKEN")

    # App Config
    target_brand: str = Field("AppleSupport", env="TARGET_BRAND")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    rag_top_k: int = Field(5, env="RAG_TOP_K")
    escalation_confidence_threshold: float = Field(0.65, env="ESCALATION_CONFIDENCE_THRESHOLD")
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Paths
    data_raw_path: str = Field("data/raw", env="DATA_RAW_PATH")
    data_processed_path: str = Field("data/processed", env="DATA_PROCESSED_PATH")
    golden_eval_path: str = Field("data/golden_eval", env="GOLDEN_EVAL_PATH")
    sqlite_db_path: str = Field("data/hiver_agent.db", env="SQLITE_DB_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
