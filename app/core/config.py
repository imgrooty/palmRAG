import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/palmmind"
    # Set True in production — keeps connections alive across Render's idle timeout
    db_pool_pre_ping: bool = True
    db_pool_size: int = 5
    db_max_overflow: int = 10
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    embedding_model: str = "all-MiniLM-L6-v2"
    groq_chat_model: str = "groq/compound"
    groq_fast_model: str = "groq/compound-mini"
    groq_whisper_model: str = "whisper-large-v3-turbo"

    max_upload_mb: int = 50
    env: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """Normalize Render/Heroku-style postgres:// URLs to asyncpg driver URLs."""
        if not v:
            return v
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        # asyncpg does NOT support libpq query-string params like ?sslmode=require.
        # SSL is handled via connect_args instead. Strip any such params here.
        if "?" in v:
            base, params = v.split("?", 1)
            # Keep only params that asyncpg's URL parser understands (none currently).
            warnings.warn(
                f"Stripped unsupported query params from DATABASE_URL: ?{params}. "
                "SSL is configured via connect_args in database.py.",
                stacklevel=2,
            )
            v = base
        if "localhost" in v or "127.0.0.1" in v:
            warnings.warn(
                "DATABASE_URL points to localhost — ensure the correct env var "
                "is set on your deployment platform (Render: 'DATABASE_URL').",
                stacklevel=2,
            )
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
