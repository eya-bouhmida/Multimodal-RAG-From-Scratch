from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    collection_name: str = "medlens"
    image_collection_name: str = "medlens_images"
    groq_model: str = "openai/gpt-oss-20b"
    cors_origins_raw: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")
    enable_bm25: bool = Field(
        default=True,
        alias="ENABLE_BM25",
        description="Set to false on memory-constrained deployments to skip building the BM25 "
        "index and fall back to dense-only search (see claude.md for why).",
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


settings = Settings()
