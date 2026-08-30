from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    groq_api_key: str
    groq_chat_model: str = "qwen/qwen3.6-27b"

    # Paths
    data_dir: Path = Path("data")
    persist_dir: Path = Path("chroma_db")

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 8

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()