"""
jarvis/ai/personality.py

Defines JARVIS's personality as a system prompt, plus the runtime
context injection JARVIS needs to reason accurately about its own
capabilities and the current environment.

Design goals:
    - Personality is entirely prompt-driven. No hardcoded canned
      responses live here or anywhere else -- this module only ever
      produces the *instructions* the model reasons from.
    - The system prompt explicitly enumerates what JARVIS CAN and
      CANNOT do, sourced dynamically from the registered tool list
      (see core/tool_selector.py), so the model is never left to
      guess or hallucinate capabilities that don't exist.
    - Supports lightweight runtime personalization (user name, time
      of day, known long-term facts from memory/long_term.py) without
      requiring a full prompt rewrite per session.

This module has no external dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ai.llm_client import ChatMessage

__all__ = [
    "PersonalityConfig",
    "PersonalityEngine",
]

logger = logging.getLogger("jarvis.ai.personality")


_BASE_SYSTEM_PROMPT = """You are JARVIS, a professional, intelligent, and calm AI assistant running locally on the user's Windows desktop.

CORE TRAITS
- Professional: You communicate clearly and respectfully, without slang, excessive casualness, or forced humor.
- Precise: You give concise, useful answers. You do not pad responses with unnecessary filler.
- Honest: You never fabricate information. If you do not know something, or are not certain, you say so directly.
- Calm: You do not editorialize, panic, or overreact, even when reporting errors or failures.
- Helpful: You proactively identify what the user needs, but you do not take irreversible or sensitive actions without going through the proper confirmation process.

BEHAVIORAL RULES
1. Never claim to have performed an action you have not actually performed. If a tool call failed or was denied, report that plainly.
2. Never pretend to have a capability that is not in your registered tool list for this session.
3. When you are uncertain about a fact, say so explicitly rather than guessing with confidence.
4. Do not use excessive enthusiasm, emoji, or jokes. Occasional light professionalism is acceptable; do not be childish.
5. When a requested action is destructive, irreversible, or sensitive (deleting files, sending emails, entering credentials, making purchases, changing system settings), you must route it through the permission and confirmation system rather than assuming consent.
6. Keep spoken responses concise by default, since they will be converted to speech. Favor a few clear sentences over long written-style paragraphs unless the user asks for detail.
7. If a user's request is ambiguous, ask a single clarifying question rather than guessing at high-stakes intent.

You are not a generic chatbot. You are a desktop assistant with real tool access on this machine, operating under a strict security and permissions layer. Treat your tool access as a responsibility, not a convenience."""


@dataclass
class PersonalityConfig:
    """Tunable parameters for JARVIS's personality prompt.

    Attributes:
        assistant_name: The name JARVIS refers to itself as. Defaults
            to "JARVIS" but kept configurable since the wake word and
            display name may diverge in future (e.g. renamed builds).
        user_name: The user's preferred name/address, if known from
            long-term memory. None if not yet established.
        verbosity: One of "concise", "standard", "detailed" -- adjusts
            the response-length guidance given to the model.
        available_tools: List of tool identifiers currently registered
            in tool_selector.py, injected so the model has an accurate
            picture of what it can and cannot do this session.
        known_facts: Short list of relevant long-term memory facts
            (e.g. "User's default browser is Chrome") to ground
            responses. Sourced from memory/long_term.py.
    """

    assistant_name: str = "JARVIS"
    user_name: Optional[str] = None
    verbosity: str = "concise"
    available_tools: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)


class PersonalityEngine:
    """Builds the system-level ChatMessage that grounds every LLM call.

    Usage:
        engine = PersonalityEngine(
            config=PersonalityConfig(
                user_name="Alex",
                available_tools=["file.read", "browser.search"],
            )
        )
        system_message = engine.build_system_message()
        messages = [system_message, ChatMessage(role="user", content=user_input)]
        response = llm_client.chat(messages)
    """

    _VERBOSITY_GUIDANCE = {
        "concise": "Keep responses to 1-3 sentences unless the user explicitly asks for more detail.",
        "standard": "Give complete answers, typically a short paragraph, without unnecessary elaboration.",
        "detailed": "Provide thorough, well-structured answers, including relevant context and reasoning.",
    }

    def __init__(self, config: Optional[PersonalityConfig] = None) -> None:
        """Initialize the personality engine.

        Args:
            config: Personality configuration. If omitted, a default
                config with no user name and no known tools is used --
                callers should update `self.config` once tool
                registration and memory retrieval have run at startup.
        """
        self.config = config or PersonalityConfig()

    def build_system_message(self) -> ChatMessage:
        """Construct the full system prompt as a ChatMessage.

        Returns:
            A ChatMessage with role="system" combining the base
            personality prompt, verbosity guidance, current runtime
            context (date/time), the accurate tool capability list,
            and any known long-term facts about the user.
        """
        sections = [_BASE_SYSTEM_PROMPT]

        verbosity_note = self._VERBOSITY_GUIDANCE.get(
            self.config.verbosity, self._VERBOSITY_GUIDANCE["concise"]
        )
        sections.append(f"\nRESPONSE LENGTH\n{verbosity_note}")

        sections.append(self._build_runtime_context_section())

        if self.config.available_tools:
            tool_list = "\n".join(f"- {tool}" for tool in sorted(self.config.available_tools))
            sections.append(
                "\nAVAILABLE CAPABILITIES\n"
                "You have access to exactly the following tools this session. "
                "Do not claim or attempt any capability not in this list:\n"
                f"{tool_list}"
            )
        else:
            sections.append(
                "\nAVAILABLE CAPABILITIES\n"
                "No automation tools are currently registered for this session. "
                "You may only hold a conversation; you cannot perform any desktop, "
                "file, browser, or system actions right now."
            )

        if self.config.known_facts:
            facts_list = "\n".join(f"- {fact}" for fact in self.config.known_facts)
            sections.append(f"\nKNOWN CONTEXT ABOUT THE USER\n{facts_list}")

        if self.config.user_name:
            sections.append(
                f"\nThe user's name is {self.config.user_name}. "
                "Address them by name occasionally, but not in every response."
            )

        full_prompt = "\n".join(sections)
        return ChatMessage(role="system", content=full_prompt)

    def _build_runtime_context_section(self) -> str:
        """Build the block giving the model accurate current date/time.

        Local LLMs have no innate sense of "now" -- without this, the
        model will confidently reason about the wrong date, which is
        exactly the kind of fabrication the personality rules forbid.
        """
        now = datetime.now()
        return (
            "\nCURRENT CONTEXT\n"
            f"- Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
            f"- Operating system: Windows"
        )

    def update_user_name(self, name: str) -> None:
        """Update the known user name, e.g. after long_term.py resolves it."""
        self.config.user_name = name
        logger.debug("Personality engine user_name updated to '%s'", name)

    def update_available_tools(self, tools: list[str]) -> None:
        """Sync the tool capability list, e.g. after tool_selector.py registration."""
        self.config.available_tools = list(tools)
        logger.debug("Personality engine available_tools updated: %s", tools)

    def update_known_facts(self, facts: list[str]) -> None:
        """Replace the known long-term facts injected into the prompt.

        Callers (memory/long_term.py) are responsible for keeping this
        list short and relevant -- this method does not truncate or
        rank facts itself.
        """
        self.config.known_facts = list(facts)
        logger.debug("Personality engine known_facts updated (%d facts)", len(facts))