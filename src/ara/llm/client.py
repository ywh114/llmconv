"""Thin OpenAI-compatible client wrapper with DeepSeek-specific features."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ara.config import AraSettings
from ara.llm.models import GameRole, StreamResult
from ara.utils.ansi import BLUE, END, GREEN, LIGHTGRAY
from ara.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    """Convert an API usage object into a plain dict recursively.

    ``openai`` returns pydantic models for usage; ``model_dump`` flattens
    nested objects such as ``prompt_tokens_details`` so downstream code can
    safely use ``dict.get``.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


class LLMClient:
    """Wrapper around :class:`openai.OpenAI` that handles role profiles,
    streaming, tool-call accumulation, and DeepSeek-specific options.

    :param settings: Application settings instance.
    """

    def __init__(self, settings: AraSettings) -> None:
        """Create the client from application settings.

        :param settings: Application settings instance.
        """
        self.settings = settings
        base_url = settings.api_endpoint
        if settings.strict_tools:
            base_url = base_url.rstrip('/') + '/beta'
        self.client = OpenAI(
            api_key=settings.api_key or os.environ.get('DEEPSEEK_API_KEY', ''),
            base_url=base_url,
            timeout=120.0,
            max_retries=2,
        )
        self.cancel_event: threading.Event | None = None

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
        name: str | None = None,
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
        :param name: Optional entity name (e.g., character name) for logging.
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
        sys_msg = kwargs['messages'][0] if kwargs['messages'] else {}
        sys_text = sys_msg.get('content', '') or ''
        logger.debug(
            f'LLM request: role={role.value}, name={name or ""}, '
            f'tools={len(tools or [])}, tool_choice={kwargs.get("tool_choice")}, '
            f'reasoning_effort={kwargs.get("reasoning_effort")}, '
            f'extra_body={kwargs.get("extra_body")}, stream={stream}, '
            f'sys_len={len(sys_text)}, msg_count={len(kwargs["messages"])}'
        )

        result = StreamResult()

        try:
            if stream:
                kwargs['stream_options'] = {'include_usage': True}
                response = self.client.chat.completions.create(
                    **kwargs, stream=True
                )
                for chunk in response:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise RuntimeError('LLM call cancelled')
                    # Usage is reported on the final chunk.  Accessing the
                    # ``usage`` attribute on the parsed pydantic object reliably
                    # hangs with DeepSeek's streaming endpoint, so we read the
                    # raw dumped dict instead.
                    raw_chunk = chunk.model_dump()
                    usage = raw_chunk.get('usage')
                    is_final = (
                        not chunk.choices
                        or chunk.choices[0].finish_reason is not None
                    )
                    if is_final and usage:
                        result.usage = _normalize_usage(usage)
                    if not chunk.choices:
                        continue
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
                        # Stream content to stderr alongside reasoning so that
                        # stdout stays clean for structured callers (agent server,
                        # tests, etc.).  CLI front-ends can print the returned
                        # result.content to stdout if they want.
                        if print_stream and role != GameRole.ORCHESTRATOR:
                            print(GREEN + delta.content + END, end='', flush=True, file=sys.stderr)
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
                if getattr(response, 'usage', None):
                    result.usage = _normalize_usage(response.usage)
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

        if print_stream and result.content and role != GameRole.ORCHESTRATOR:
            # Echo the full response to stderr for visibility in CLI/TUI usage.
            print(LIGHTGRAY + result.content + END, file=sys.stderr)

        usage = result.usage
        prompt_tokens = usage.get('prompt_tokens') if usage else None
        completion_tokens = usage.get('completion_tokens') if usage else None
        cached_tokens = None
        if usage:
            cached_tokens = (
                usage.get('prompt_tokens_details', {}).get('cached_tokens')
                or usage.get('prompt_cache_hit_tokens')
            )
        tool_summary = ', '.join(
            f"{tc['function']['name']}({tc['function']['arguments'][:80]!r}...)"
            for tc in result.tool_calls
        )
        logger.debug(
            f'LLM response: content_len={len(result.content)}, '
            f'reasoning_len={len(result.reasoning_content)}, '
            f'tool_calls={len(result.tool_calls)}, '
            f'prompt_tokens={prompt_tokens}, '
            f'completion_tokens={completion_tokens}, '
            f'cached_tokens={cached_tokens}, '
            f'tool_summary=[{tool_summary}]'
        )
        log_prefix = f'{role.value}:{name}' if name else role.value
        # Log reasoning first, then the spoken/written content.
        if result.reasoning_content:
            logger.info(f'{log_prefix}:reasoning:{result.reasoning_content!r}')
        if result.content or not result.reasoning_content:
            logger.info(f'{log_prefix}:content:{result.content!r}')
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
            logger.debug(f'Sub-agent request: model={kwargs.get("model")}, max_tokens={kwargs.get("max_tokens")}')
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            logger.debug(f'Sub-agent response: content_len={len(content)}')
            logger.info(f'subagent:content:{content!r}')
            return content
        except Exception as exc:
            logger.warning(f"Sub-agent call failed: {exc}")
            return ""
