from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    GOOGLE_CLOUD_PROJECT_ID: str | None = None


settings = Settings()
