"""Environment-driven configuration (PROJECT.md section 12).

Every value comes from the environment. Nothing here is ever logged or echoed in
a response - the API key in particular must not leave this process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider (wired up in P2; declared here so ops can set it once) ---
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_fallback_model: str = Field(default="openai/gpt-oss-20b")
    llm_timeout_seconds: float = Field(default=8.0)
    llm_max_retries: int = Field(default=2)
    #: Wall-clock ceiling for the whole interpretation stage. Without it, every
    #: attempt hanging to its own timeout could exceed the judge's 30s per-request
    #: limit; once this is spent we stop calling the provider and interpret
    #: deterministically instead.
    llm_total_budget_seconds: float = Field(default=12.0)

    # --- service ---
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)

    def safe_dump(self) -> dict[str, object]:
        """Config snapshot with no secret values - safe to log."""
        return {
            "groq_base_url": self.groq_base_url,
            "groq_model": self.groq_model,
            "groq_fallback_model": self.groq_fallback_model,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_retries": self.llm_max_retries,
            "llm_total_budget_seconds": self.llm_total_budget_seconds,
            "llm_configured": self.llm_configured,
            "port": self.port,
            "log_level": self.log_level,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
