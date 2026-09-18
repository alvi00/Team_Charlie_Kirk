"""Environment-driven configuration (PROJECT.md section 12).

Every value comes from the environment. Nothing here is ever logged or echoed in
a response - the API key in particular must not leave this process.

The LLM layer is provider-agnostic: it speaks the OpenAI Chat Completions API,
which OpenAI, Groq, Together, Fireworks and others all expose. OpenAI is the
primary provider; an optional second provider can be configured as a
cross-provider failover so a single vendor outage cannot take interpretation down.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- primary provider: OpenAI ---
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-5.6-luna")
    #: second model at the same provider, tried if the primary errors
    openai_fallback_model: str = Field(default="gpt-4.1-mini")

    # --- optional cross-provider failover (e.g. Groq) ---
    # Leave the key empty to disable. Configured separately from the primary so a
    # single vendor outage or quota exhaustion cannot take interpretation down.
    fallback_provider_api_key: str = Field(default="")
    fallback_provider_base_url: str = Field(default="https://api.groq.com/openai/v1")
    fallback_provider_model: str = Field(default="openai/gpt-oss-120b")

    # --- call behaviour ---
    llm_timeout_seconds: float = Field(default=8.0)
    llm_max_retries: int = Field(default=2)
    #: Wall-clock ceiling for the whole interpretation stage. Without it, every
    #: attempt hanging to its own timeout could exceed the judge's 30s per-request
    #: limit; once this is spent we stop calling providers and interpret
    #: deterministically instead.
    llm_total_budget_seconds: float = Field(default=12.0)
    #: Reasoning models accept this; it is dropped automatically if the model
    #: rejects it. Keep it low - the rules are all in the prompt.
    llm_reasoning_effort: str = Field(default="low")

    # --- service ---
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or self.fallback_provider_api_key)

    @property
    def failover_configured(self) -> bool:
        return bool(self.fallback_provider_api_key)

    def safe_dump(self) -> dict[str, object]:
        """Config snapshot with no secret values - safe to log."""
        return {
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "openai_fallback_model": self.openai_fallback_model,
            "openai_key_present": bool(self.openai_api_key),
            "failover_base_url": self.fallback_provider_base_url
            if self.failover_configured
            else None,
            "failover_model": self.fallback_provider_model
            if self.failover_configured
            else None,
            "failover_configured": self.failover_configured,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_retries": self.llm_max_retries,
            "llm_total_budget_seconds": self.llm_total_budget_seconds,
            "llm_reasoning_effort": self.llm_reasoning_effort,
            "port": self.port,
            "log_level": self.log_level,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
