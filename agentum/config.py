from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for the Agentum framework.

    Loads settings from a .env file or environment variables, providing
    a single source of truth for all API keys and other configurations.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM Provider API Keys ---
    GOOGLE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # --- Reranker API Keys ---
    COHERE_API_KEY: str | None = None

    # --- Tool-Specific API Keys & Config ---
    TAVILY_API_KEY: str | None = None
    GOOGLE_CLOUD_PROJECT_ID: str | None = None


# Create a singleton instance to be used throughout the application
settings = Settings()
