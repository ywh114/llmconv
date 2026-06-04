"""Application settings loaded from ``pyproject.toml`` or environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PyprojectTomlConfigSettingsSource


class AraSettings(BaseSettings):
    """Single consolidated settings class for the Ara engine.

    Values are read from the ``[tool.ara]`` table in ``pyproject.toml``
    and may be overridden by environment variables prefixed with ``ARA_``.

    :ivar api_key: API key for the LLM provider.  Falls back to the
        ``DEEPSEEK_API_KEY`` environment variable if empty.
    :ivar api_endpoint: Base URL for the LLM API.
    :ivar api_model: Model identifier sent to the provider.
    :ivar embedding_model: ``sentence-transformers`` model name used by ChromaDB.
    :ivar data_dir: Root directory for persistent data (ChromaDB, assets, etc.).
    :ivar language: Default language for generated text.
    :ivar temperature_character: Sampling temperature for character turns.
    :ivar temperature_narrator: Sampling temperature for narrator turns.
    :ivar temperature_orchestrator: Sampling temperature for orchestrator turns.
    :ivar strict_tools: When ``True``, request the DeepSeek beta endpoint
        and emit ``strict: true`` in tool schemas for guaranteed JSON schema
        compliance.
    """

    model_config = SettingsConfigDict(
        env_prefix='ARA_',
        pyproject_toml_table_header=('tool', 'ara'),
        pyproject_toml_depth=2,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    api_key: str = ''
    api_endpoint: str = 'https://api.deepseek.com'
    api_model: str = 'deepseek-v4-pro'
    embedding_model: str = 'all-MiniLM-L6-v2'
    data_dir: Path = Path('data')
    language: str = 'English'
    temperature_character: float = 0.9
    temperature_narrator: float = 0.6
    temperature_orchestrator: float = 0.6
    temperature_summarizer: float = 0.4
    strict_tools: bool = False

    @property
    def chroma_path(self) -> Path:
        """Resolved path to the ChromaDB persistent storage directory.

        :return: ``data_dir / "chroma"``
        """
        return self.data_dir / 'chroma'
