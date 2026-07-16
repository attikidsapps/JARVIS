"""
jarvis/ai/llm_client.py

Thin, typed wrapper around the local Ollama server running the
dolphin-llama3 model. This is the single point of contact between
JARVIS's reasoning/planning layers and the underlying LLM.

Design goals:
    - Local-first, zero cost: talks to Ollama's local HTTP API
      (default http://localhost:11434). No cloud API keys, no
      per-token billing.
    - No hardcoded responses. All reasoning, planning, and
      conversational output must come from the model.
    - Resilient to Ollama's cold-start latency (the first request
      after `ollama serve` starts, or after model eviction from
      memory, can take several seconds to load the model).
    - Supports both plain conversational chat and structured
      (JSON-constrained) output for the planner/tool-selector stages.

API compatibility note:
    This file targets ollama==0.6.2 (pinned in requirements.txt).
    The ollama Python client >=0.4.0 returns typed Pydantic response
    objects (ListResponse, ChatResponse), NOT plain dicts. All
    attribute access uses the Pydantic model API (response.message.content,
    response.done, etc.), not dict.get(). Do not revert to dict access.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

try:
    import ollama
    from ollama import ResponseError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'ollama' package is required. Install with: pip install ollama\n"
        "Also ensure the Ollama application is installed and running locally "
        "(https://ollama.com), and that the model has been pulled via: "
        "ollama pull dolphin-llama3"
    ) from exc

__all__ = [
    "LLMClientError",
    "ModelUnavailableError",
    "ChatMessage",
    "LLMResponse",
    "OllamaClient",
]

logger = logging.getLogger("jarvis.ai.llm_client")

DEFAULT_MODEL = "dolphin-llama3"
DEFAULT_HOST = "http://localhost:11434"


class LLMClientError(Exception):
    """Base exception for all LLM client failures."""


class ModelUnavailableError(LLMClientError):
    """Raised when Ollama is unreachable or the model is not pulled locally."""


@dataclass
class ChatMessage:
    """A single message in a conversation.

    Attributes:
        role: One of "system", "user", "assistant", or "tool".
        content: The message text.
        name: Optional identifier, used for tool-role messages to
            indicate which tool produced the content.
    """

    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass
class LLMResponse:
    """Normalized response from a chat call.

    Attributes:
        content: The full text of the model's reply.
        model: The model name that generated this response.
        done: Whether generation completed (vs. truncated).
        total_duration_ms: Total wall-clock time for the call, in
            milliseconds, if reported by Ollama.
        raw: The raw Pydantic ChatResponse object returned by the
            ollama client, for callers that need fields not surfaced
            here (e.g. token counts). Access via attribute, not dict.
    """

    content: str
    model: str
    done: bool
    total_duration_ms: Optional[float] = None
    raw: Any = field(default=None)


class OllamaClient:
    """Client for interacting with a local Ollama server.

    Usage:
        client = OllamaClient()
        response = client.chat([
            ChatMessage(role="system", content=JARVIS_SYSTEM_PROMPT),
            ChatMessage(role="user", content="What's on my calendar today?"),
        ])
        print(response.content)

    For planner output that must be valid JSON:
        plan = client.chat_json(
            messages=[...],
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        """Initialize the Ollama client wrapper.

        Args:
            model: Ollama model tag to use for all calls. Must already
                be pulled locally (`ollama pull dolphin-llama3`).
            host: Base URL of the local Ollama server.
            max_retries: Number of attempts for transient connection
                failures before raising ModelUnavailableError.
            retry_backoff_seconds: Base delay between retries;
                multiplied by attempt number for simple linear backoff.
            request_timeout_seconds: Per-request timeout passed to the
                underlying ollama client.
        """
        self.model = model
        self.host = host
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = ollama.Client(host=host, timeout=request_timeout_seconds)

        self._verify_model_available()

    # ------------------------------------------------------------------
    # Startup verification
    # ------------------------------------------------------------------

    def _verify_model_available(self) -> None:
        """Confirm Ollama is reachable and the target model is pulled.

        Uses the Pydantic ListResponse API (ollama>=0.4.0):
            response.models  → list[ModelInfo]
            model_info.model → str (e.g. "dolphin-llama3:latest")

        Raises:
            ModelUnavailableError: If the Ollama server cannot be
                reached, or the model is not present locally.
        """
        try:
            # ollama>=0.4.0: returns ListResponse, not a plain dict.
            list_response = self._client.list()
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Could not reach Ollama server at {self.host}. "
                "Ensure the Ollama application is running (it must be "
                "started separately from JARVIS, e.g. via the Ollama "
                "desktop app or `ollama serve`)."
            ) from exc

        # list_response.models is a list of ModelInfo Pydantic objects.
        # Each has a .model attribute (the full tag, e.g. "dolphin-llama3:latest")
        # and a .name attribute (alias for .model in some versions).
        model_names: set[str] = set()
        for entry in list_response.models:
            # .model is the canonical attribute in ollama>=0.4.0
            tag = getattr(entry, "model", None) or getattr(entry, "name", "") or ""
            model_names.add(tag)

        # Match on prefix to handle ":latest" suffix variations.
        found = any(
            name == self.model or name.startswith(f"{self.model}:")
            for name in model_names
        )
        if not found:
            raise ModelUnavailableError(
                f"Model '{self.model}' is not available locally. "
                f"Pull it first with: ollama pull {self.model}"
            )

        logger.info("Ollama client initialized: model='%s', host='%s'", self.model, self.host)

    # ------------------------------------------------------------------
    # Core chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request to dolphin-llama3.

        Uses the Pydantic ChatResponse API (ollama>=0.4.0):
            result.message.content  → str
            result.done             → bool
            result.model            → str
            result.total_duration   → int (nanoseconds, may be None)

        Args:
            messages: Ordered conversation history, oldest first.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Optional cap on generated tokens (maps to
                Ollama's `num_predict`). None lets the model decide.
            format_json: If True, instructs Ollama to constrain output
                to valid JSON. Use for planner/tool-selection calls.

        Returns:
            LLMResponse containing the assistant's reply.

        Raises:
            ModelUnavailableError: If all retry attempts fail.
        """
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload_messages = [m.to_dict() for m in messages]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.monotonic()
                # ollama>=0.4.0: returns ChatResponse (Pydantic model), not dict.
                result = self._client.chat(
                    model=self.model,
                    messages=payload_messages,
                    options=options,
                    format="json" if format_json else "",
                    stream=False,
                )
                elapsed_ms = (time.monotonic() - start) * 1000

                # Access Pydantic attributes, not dict keys.
                content: str = result.message.content or ""
                model_name: str = result.model or self.model
                done: bool = bool(result.done)

                # total_duration is in nanoseconds; convert to milliseconds.
                total_ns: Optional[int] = getattr(result, "total_duration", None)
                total_ms = (total_ns / 1_000_000) if total_ns is not None else elapsed_ms

                return LLMResponse(
                    content=content,
                    model=model_name,
                    done=done,
                    total_duration_ms=total_ms,
                    raw=result,  # Pydantic ChatResponse — access via attributes
                )

            except ResponseError as exc:
                # Model-level errors are not transient.
                logger.error("Ollama returned a response error: %s", exc)
                raise ModelUnavailableError(f"Ollama response error: {exc}") from exc

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Chat request failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        raise ModelUnavailableError(
            f"Failed to get a response from Ollama after {self.max_retries} attempts."
        ) from last_error

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """Stream a chat completion token-by-token for low-latency TTS.

        Yields incremental content chunks as they arrive from Ollama.
        Each chunk is a ChatResponse object (ollama>=0.4.0); content is
        accessed via chunk.message.content (not dict-style .get()).

        Args:
            messages: Ordered conversation history.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Yields:
            str: Incremental content chunks (may be empty strings on
                 heartbeat chunks — callers should filter falsy values).

        Raises:
            ModelUnavailableError: If the stream fails to start or is
                interrupted by a connection error.
        """
        payload_messages = [m.to_dict() for m in messages]
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}

        try:
            stream = self._client.chat(
                model=self.model,
                messages=payload_messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                # ollama>=0.4.0: each chunk is a ChatResponse Pydantic object.
                # chunk.message is a Message object; .content is the text delta.
                content: str = chunk.message.content or ""
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            logger.error("Streaming chat failed: %s", exc)
            raise ModelUnavailableError(f"Streaming chat failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Structured output for planning / tool selection
    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.2,
        max_retries_on_parse_failure: int = 2,
    ) -> dict[str, Any]:
        """Send a chat request and parse the reply as JSON.

        Intended for core/planner.py and core/tool_selector.py, where
        the model must return a structured plan or tool-call payload
        rather than free-form conversational text.

        Args:
            messages: Conversation history. The final system or user
                message should explicitly instruct the model to
                respond with JSON matching the expected schema.
            temperature: Sampling temperature (low default for
                structured output).
            max_retries_on_parse_failure: Retry count on invalid JSON.

        Returns:
            The parsed JSON object as a dict.

        Raises:
            LLMClientError: If the model fails to produce valid JSON
                after all retries.
            ModelUnavailableError: If the underlying chat call fails.
        """
        working_messages = list(messages)

        for attempt in range(max_retries_on_parse_failure + 1):
            response = self.chat(working_messages, temperature=temperature, format_json=True)
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Model returned invalid JSON on attempt %d/%d: %s",
                    attempt + 1, max_retries_on_parse_failure + 1, exc,
                )
                if attempt < max_retries_on_parse_failure:
                    working_messages = working_messages + [
                        ChatMessage(role="assistant", content=response.content),
                        ChatMessage(
                            role="user",
                            content=(
                                "That was not valid JSON. Respond again with "
                                "ONLY a valid JSON object and no other text."
                            ),
                        ),
                    ]
                else:
                    raise LLMClientError(
                        f"Model failed to produce valid JSON after "
                        f"{max_retries_on_parse_failure + 1} attempts. "
                        f"Last output: {response.content[:500]!r}"
                    ) from exc

        # Unreachable, but satisfies type checkers.
        raise LLMClientError("chat_json exhausted retries without returning or raising.")