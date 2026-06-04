"""Thin OpenAI-compatible client wrapper with DeepSeek-specific features."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ara.config import AraSettings
from ara.models import GameRole, StreamResult
from ara.utils.ansi import BLUE, END, GREEN, LIGHTGRAY
from ara.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Wrapper around :class:`openai.OpenAI` that handles role profiles,
    streaming, tool-call accumulation, and DeepSeek-specific options.

    :param settings: Application settings instance.
    """

    def __init__(self, settings: AraSettings) -> None:
        self.settings = settings
        base_url = settings.api_endpoint
        if settings.strict_tools:
            base_url = base_url.rstrip('/') + '/beta'
        self.client = OpenAI(
            api_key=settings.api_key or os.environ.get('DEEPSEEK_API_KEY', ''),
            base_url=base_url,
        )

    def _profile_kwargs(self, role: GameRole) -> dict[str, Any]:
        """Return the base keyword arguments for a given *role*.

        :param role: The game role whose temperature profile should be used.
        :return: Dictionary with ``model``, ``temperature``, and ``max_tokens``.
        """
        temps = {
            GameRole.CHARACTER: self.settings.temperature_character,
            GameRole.NARRATOR: self.settings.temperature_narrator,
            GameRole.ORCHESTRATOR: self.settings.temperature_orchestrator,
            GameRole.SUMMARIZER: self.settings.temperature_summarizer,
        }
        kwargs: dict[str, Any] = {
            'model': self.settings.api_model,
            'temperature': temps[role],
            'max_tokens': 32768,
        }
        if role in (GameRole.CHARACTER, GameRole.NARRATOR, GameRole.SUMMARIZER):
            kwargs['frequency_penalty'] = 1.0
        return kwargs

    def complete(
        self,
        role: GameRole,
        system_prompt: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        stream: bool = True,
        print_stream: bool = False,
    ) -> StreamResult:
        """Send a chat-completion request and return a :class:`StreamResult`.

        When *stream* is ``True`` the response is consumed incrementally so
        that reasoning content can be printed in real time.  Tool-call deltas
        are stitched together into complete call descriptors.

        :param role: Role profile controlling temperature.
        :param system_prompt: System message content.
        :param messages: Conversation history (excludes system message).
        :param tools: Optional list of tool schemas.
        :param tool_choice: Optional tool-choice directive (e.g. ``"required"``).
        :param stream: Whether to stream the response.
        :return: Accumulated result containing content, reasoning content, and
            any tool calls.
        """
        kwargs: dict[str, Any] = self._profile_kwargs(role)
        kwargs['messages'] = [
            {'role': 'system', 'content': system_prompt},
            *messages,
        ]

        # Enable DeepSeek thinking/reasoning mode for all roles.
        # The ORCHESTRATOR no longer forces tool_choice="required" because
        # thinking mode is incompatible with required tools on DeepSeek.
        # Instead, the orchestrator system prompt instructs the model to
        # ALWAYS call the next_round tool and NEVER output free text.
        kwargs['reasoning_effort'] = 'high'
        kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}

        # Defensive: ensure every message has a valid string content.
        # DeepSeek thinking mode may produce content=None on assistant
        # messages that only contain reasoning_content.
        for m in kwargs['messages']:
            if not isinstance(m.get('content'), str):
                m['content'] = str(m.get('content', ''))

        if tools:
            kwargs['tools'] = tools
        # Apply caller-supplied tool_choice only (no automatic override).
        if tool_choice:
            kwargs['tool_choice'] = tool_choice

        logger.debug(
            f'LLM request: role={role.value}, tools={len(tools or [])}, '
            f'tool_choice={kwargs.get("tool_choice")}, '
            f'reasoning_effort={kwargs.get("reasoning_effort")}, '
            f'extra_body={kwargs.get("extra_body")}, stream={stream}'
        )

        result = StreamResult()

        try:
            if stream:
                response = self.client.chat.completions.create(
                    **kwargs, stream=True
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if getattr(delta, 'reasoning_content', None):
                        result.reasoning_content += delta.reasoning_content
                        if print_stream:
                            # Reasoning goes to stderr so that stdout captures
                            # only the final spoken content (used by the web API).
                            print(
                                BLUE + delta.reasoning_content + END,
                                end='',
                                flush=True,
                                file=sys.stderr,
                            )
                    if delta.content:
                        result.content += delta.content
                        # Only stream content to stdout for character/narrator roles.
                        # Orchestrator output must not pollute stdout because the
                        # agent server captures stdout to build the narrator/character
                        # dialogue payload.
                        if print_stream and role != GameRole.ORCHESTRATOR:
                            print(GREEN + delta.content + END, end='', flush=True)
                    if getattr(delta, 'tool_calls', None):
                        for tc in delta.tool_calls:
                            idx = tc.index
                            while len(result.tool_calls) <= idx:
                                result.tool_calls.append(
                                    {
                                        'id': '',
                                        'function': {'name': '', 'arguments': ''},
                                        'type': 'function',
                                    }
                                )
                            if tc.id:
                                result.tool_calls[idx]['id'] = tc.id
                            if tc.function and tc.function.name:
                                result.tool_calls[idx]['function']['name'] = (
                                    tc.function.name
                                )
                            if tc.function and tc.function.arguments:
                                result.tool_calls[idx]['function']['arguments'] += (
                                    tc.function.arguments
                                )
            else:
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                result.content = msg.content or ''
                result.reasoning_content = (
                    getattr(msg, 'reasoning_content', '') or ''
                )
                if msg.tool_calls:
                    result.tool_calls = [
                        {
                            'id': tc.id,
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            },
                            'type': 'function',
                        }
                        for tc in msg.tool_calls
                    ]
        except Exception as exc:
            logger.error(f'LLM request failed: {exc}')
            raise

        if print_stream and result.content:
            # Print the full response to stderr as well so it is visible
            # even when stdout is being captured (e.g. by the agent server).
            print(LIGHTGRAY + result.content + END, file=sys.stderr)

        logger.debug(
            f'LLM response: content_len={len(result.content)}, '
            f'reasoning_len={len(result.reasoning_content)}, '
            f'tool_calls={len(result.tool_calls)}'
        )
        return result

    def complete_subagent(
        self,
        task: str,
        context: str,
        system_prompt: str = "You are a focused sub-agent. Complete the task concisely.",
        max_tokens: int = 512,
    ) -> str:
        """Run a focused sub-agent task and return the text output.

        This is a simplified non-streaming completion intended for
        summarisation, reflection, and other auxiliary tasks.

        :param task: The instruction for the sub-agent.
        :param context: Supporting context to ground the task.
        :param system_prompt: Optional override for the system prompt.
        :param max_tokens: Cap on output length.
        :return: Generated text (empty string on failure).
        """
        kwargs: dict[str, Any] = {
            'model': self.settings.api_model,
            'temperature': 0.4,
            'max_tokens': max_tokens,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Task: {task}\n\nContext:\n{context}"},
            ],
            'extra_body': {'thinking': {'type': 'disabled'}},
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"Sub-agent call failed: {exc}")
            return ""
