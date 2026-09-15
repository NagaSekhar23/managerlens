from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/ -> managerlens/ (project root, where .env lives in local dev).
# In production this file simply won't exist and settings come from real environment
# variables injected by the hosting platform — pydantic-settings handles a missing
# env_file gracefully.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / the root .env.

    Every setting has a safe local-development default so the app runs out of the box
    with `docker compose up`, but nothing here is a real secret — GEMINI_API_KEY has no
    default and must be supplied by the environment in every deployment.
    """

    app_name: str = "ManagerLens API"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql://managerlens:managerlens@localhost:5432/managerlens"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    # Applies to every outbound Gemini call (analysis, embedding, evaluation judge) so a
    # slow/unresponsive upstream can't hang a request indefinitely.
    gemini_timeout_seconds: float = 30.0

    # A JSON array in the environment, e.g. CORS_ORIGINS=["https://managerlens.example.com"].
    # Never defaults to "*" — an explicit allowlist is required for any non-local deployment.
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


settings = Settings()
