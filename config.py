from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required — app won't start if missing
    mongodb_uri: str
    openai_api_key: str
    api_key: str

    # Optional with defaults
    redis_url: str = "redis://redis:6379/0"
    faiss_index_path: str = "faiss_index"
    llm_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    log_dir: str = "logs"
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
