"""Tests for the LLM client wrapper."""

from __future__ import annotations

from unittest.mock import patch

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.models import GameRole


class _FakeUsage:
    """Minimal stand-in for an OpenAI CompletionUsage object."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, cached: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakeDetails(cached)

    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens_details": self.prompt_tokens_details.model_dump(),
        }


class _FakeDetails:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached

    def model_dump(self) -> dict:
        return {"cached_tokens": self.cached_tokens}


class _FakeMessage:
    def __init__(self) -> None:
        self.content = "hello"
        self.reasoning_content = ""
        self.tool_calls = None


class _FakeChoice:
    def __init__(self) -> None:
        self.message = _FakeMessage()
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self) -> None:
        self.usage = _FakeUsage(10, 5, 3)
        self.choices = [_FakeChoice()]


def test_complete_normalizes_nested_usage_objects() -> None:
    """Regression test: nested pydantic usage objects must not crash .get()."""
    settings = AraSettings(api_key="", api_endpoint="", api_model="")
    client = LLMClient(settings)

    fake_response = _FakeResponse()
    with patch.object(
        client.client.chat.completions, "create", return_value=fake_response
    ):
        result = client.complete(
            role=GameRole.CHARACTER,
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )

    assert result.usage is not None
    assert result.usage.get("prompt_tokens") == 10
    assert result.usage.get("completion_tokens") == 5
    # The important part: nested PromptTokensDetails became a plain dict.
    details = result.usage.get("prompt_tokens_details")
    assert isinstance(details, dict)
    assert details.get("cached_tokens") == 3
