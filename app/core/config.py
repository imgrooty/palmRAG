from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "palmmind"

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

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
