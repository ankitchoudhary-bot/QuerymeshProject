"""Centralized application configuration.

QueryMesh rule: "Configuration must come from environment/config objects."
This module is the single source of truth for configuration. Nothing else
in the codebase should read from `os.environ` directly — everything should
depend on the `Settings` object (or the `get_settings()` accessor) exposed
here.

Phase 0 only defines the *shape* of configuration (app-level fields plus
placeholders for future phases). Fields for OpenSearch, Jina, Redis,
Langfuse, Logfire, MCP, and A2A already exist as placeholders because the
`.env.example` documents them, but most are optional strings until the
phase that implements them starts validating/using them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App -----------------------------------------------------------
    qm_env: Literal["development", "test", "production"] = Field(
        default="development", alias="QM_ENV"
    )
    qm_log_level: str = Field(default="INFO", alias="QM_LOG_LEVEL")
    qm_app_name: str = Field(default="querymesh", alias="QM_APP_NAME")

    # --- LLM (wired up starting Phase 1) --------------------------------
    llm_provider: str | None = Field(default=None, alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")

    # --- OpenSearch (wired up starting Phase 5) -------------------------
    opensearch_url: str | None = Field(default=None, alias="OPENSEARCH_URL")
    opensearch_username: str | None = Field(default=None, alias="OPENSEARCH_USERNAME")
    opensearch_password: str | None = Field(default=None, alias="OPENSEARCH_PASSWORD")
    opensearch_index: str = Field(default="querymesh-papers", alias="OPENSEARCH_INDEX")

    # --- Jina embeddings (wired up starting Phase 7) ---------------------
    jina_api_key: str | None = Field(default=None, alias="JINA_API_KEY")
    jina_embedding_model: str | None = Field(default=None, alias="JINA_EMBEDDING_MODEL")

    # --- Redis / Upstash (wired up starting Phase 19) ---------------------
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    redis_ttl_seconds: int = Field(default=3600, alias="REDIS_TTL_SECONDS")

    # --- Langfuse (wired up starting Phase 18) ----------------------------
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")

    # --- Logfire (wired up starting Phase 18) ------------------------------
    logfire_token: str | None = Field(default=None, alias="LOGFIRE_TOKEN")

    # --- arXiv (wired up starting Phase 2) --------------------------------
    arxiv_base_url: str = Field(
        default="http://export.arxiv.org/api/query", alias="ARXIV_BASE_URL"
    )
    arxiv_max_results: int = Field(default=20, alias="ARXIV_MAX_RESULTS")
    arxiv_rate_limit_seconds: float = Field(default=3.0, alias="ARXIV_RATE_LIMIT_SECONDS")

    # --- MCP (wired up starting Phase 13/16) -------------------------------
    mcp_host: str = Field(default="127.0.0.1", alias="MCP_HOST")
    mcp_port: int = Field(default=8765, alias="MCP_PORT")
    mcp_auth_token: str | None = Field(default=None, alias="MCP_AUTH_TOKEN")

    # --- A2A (wired up starting Phase 14/17) -------------------------------
    a2a_timeout_seconds: float = Field(default=30.0, alias="A2A_TIMEOUT_SECONDS")
    a2a_max_retries: int = Field(default=2, alias="A2A_MAX_RETRIES")

    @property
    def is_production(self) -> bool:
        return self.qm_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance.

    Cached with lru_cache so repeated calls don't re-parse the environment.
    Tests that need a fresh Settings object should call
    `get_settings.cache_clear()` first.
    """
    return Settings()
