"""A2A Toolkit — convenience factory for multiple A2A agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from .runnable import A2ARunnable

if TYPE_CHECKING:
    from a2a.types import AgentCard

logger = logging.getLogger(__name__)


class A2AToolkit:
    """Factory for connecting to multiple A2A agents at once.

    The primary output is a list of A2ARunnable instances (agent-first).
    Tools are derived via .as_tool() for LLM function-calling use cases.

    Example::

        async with A2AToolkit.from_agent_urls([
            "http://summarizer-agent:8080",
            "http://search-agent:8080",
        ]) as toolkit:
            # Agent-first: get runnables
            for agent in toolkit.get_runnables():
                result = await agent.ainvoke("hello")

            # Tool-derived: for LLM function calling
            tools = toolkit.get_tools()
            llm_with_tools = llm.bind_tools(tools)
    """

    def __init__(self) -> None:
        self._runnables: dict[str, A2ARunnable] = {}
        self._agent_cards: dict[str, AgentCard] = {}

    @classmethod
    async def from_agent_urls(
        cls,
        urls: list[str],
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> A2AToolkit:
        """Create toolkit by discovering agents at the given URLs.

        Args:
            urls: List of A2A agent base URLs.
            timeout: Request timeout for all clients.
            headers: Common headers for all requests.

        Returns:
            Configured A2AToolkit with all agents loaded.
        """
        toolkit = cls()

        for url in urls:
            try:
                runnable = await A2ARunnable.from_agent_url(
                    url, timeout=timeout, headers=headers
                )
                card = runnable.agent_card
                toolkit._runnables[card.name] = runnable
                toolkit._agent_cards[card.name] = card
                logger.info("Loaded A2A agent '%s' from %s", card.name, url)
            except Exception:
                logger.exception("Failed to load A2A agent from %s", url)
                continue

        return toolkit

    def get_runnables(self) -> list[A2ARunnable]:
        """Get all agent runnables (primary API)."""
        return list(self._runnables.values())

    def get_runnable(self, name: str) -> A2ARunnable | None:
        """Get a specific runnable by agent name."""
        return self._runnables.get(name)

    def get_tools(self) -> list[BaseTool]:
        """Get all agents as LangChain tools (one tool per skill).

        Each agent's skills are expanded into individual tools via
        ``.as_tools()``. If an agent has no skills, a single agent-level
        tool is created as fallback.
        """
        tools: list[BaseTool] = []
        for r in self._runnables.values():
            tools.extend(r.as_tools())
        return tools

    def get_agent_cards(self) -> list[AgentCard]:
        """Get all discovered agent cards."""
        return list(self._agent_cards.values())

    async def close(self) -> None:
        """Close all underlying clients."""
        for runnable in self._runnables.values():
            await runnable.close()
        self._runnables.clear()

    async def __aenter__(self) -> A2AToolkit:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
