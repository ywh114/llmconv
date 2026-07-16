"""Application settings loaded from ``pyproject.toml`` or environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
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
        """Insert ``pyproject.toml`` config between env and dotenv sources."""
        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    api_key: str = ''
    api_endpoint: str = ''
    api_model: str = ''
    embedding_model: str = 'all-MiniLM-L6-v2'
    data_dir: Path = Path(__file__).parent.parent.parent / 'data'
    language: str = 'en'
    temperature_character: float = 0.9
    temperature_narrator: float = 0.6
    temperature_orchestrator: float = 0.6
    temperature_summarizer: float = 0.4
    strict_tools: bool = False

    @model_validator(mode='after')
    def _resolve_data_dir(self):
        """Anchor relative data_dir to the project root so ChromaDB and saves
        don't wander into subdirectories based on the current working directory.
        """
        if not self.data_dir.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = (project_root / self.data_dir).resolve()
        return self

    @property
    def chroma_path(self) -> Path:
        """Resolved path to the ChromaDB persistent storage directory.

        :return: ``data_dir / "chroma"``
        """
        return self.data_dir / 'chroma'

    @property
    def saves_path(self) -> Path:
        """Resolved path to the save file directory.

        :return: ``data_dir / "saves"``
        """
        return self.data_dir / 'saves'

    @property
    def sockets_path(self) -> Path:
        """Resolved path to the UNIX socket directory.

        :return: ``data_dir / "sockets"``
        """
        return self.data_dir / 'sockets'

    @property
    def default_socket_path(self) -> Path:
        """Full path to the default agent UNIX socket.

        :return: ``sockets_path / "ara_agent.sock"``
        """
        return self.sockets_path / 'ara_agent.sock'

    @property
    def assets_path(self) -> Path:
        """Root directory for game assets.

        :return: ``data_dir / "assets"``
        """
        return self.data_dir / 'assets'

    @property
    def plot_path(self) -> Path:
        """Directory containing per-story plot TOML files.

        :return: ``assets_path / "plot"``
        """
        return self.assets_path / 'plot'

    def world_path(self, story: str | None = None) -> Path:
        """Directory containing world-setting TOML files.

        :param story: Optional story id. When given, returns the per-story
            world directory ``assets/world/<story>/``.
        :return: ``assets_path / "world"`` or ``assets_path / "world" / <story>``.
        """
        base = self.assets_path / 'world'
        return base / story if story else base

    def items_path(self, story: str | None = None) -> Path:
        """Directory containing item definition TOML files.

        :param story: Optional story id. When given, returns the per-story
            item directory ``assets/items/<story>/``.
        :return: ``assets_path / "items"`` or ``assets_path / "items" / <story>``.
        """
        base = self.assets_path / 'items'
        return base / story if story else base

    def fortune_path(self, story: str | None = None) -> Path:
        """Directory containing fortune-telling data files.

        :param story: Optional story id. When given, returns the per-story
            fortune directory ``assets/fortune/<story>/``.
        :return: ``assets_path / "fortune"`` or ``assets_path / "fortune" / <story>``.
        """
        base = self.assets_path / 'fortune'
        return base / story if story else base

    def characters_path(self, story: str | None = None) -> Path:
        """Directory for character assets.

        :param story: Optional story id. When given, returns the per-story
            character directory ``assets/cc/<story>/``.
        :return: ``assets_path / "cc"`` or ``assets_path / "cc" / <story>``.
        """
        base = self.assets_path / 'cc'
        return base / story if story else base

    def locations_path(self, story: str | None = None) -> Path:
        """Directory for location assets.

        :param story: Optional story id. When given, returns the per-story
            location directory ``assets/lc/<story>/``.
        :return: ``assets_path / "lc"`` or ``assets_path / "lc" / <story>``.
        """
        base = self.assets_path / 'lc'
        return base / story if story else base

    def anonymous_path(self, story: str | None = None) -> Path:
        """Directory for anonymous background-character sprites.

        Per-story anonymous sprites take priority over the global pool.

        :param story: Optional story id.
        :return: ``characters_path(story) / "anonymous"``.
        """
        return self.characters_path(story) / 'anonymous'
