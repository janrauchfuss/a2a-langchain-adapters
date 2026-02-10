"""A2A Runnable — primary LangChain integration for A2A agents."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tools import BaseTool, ToolException

from .auth import A2AAuthConfig
from .client_wrapper import (
    A2AClientWrapper,
    A2AInput,
)
from .exceptions import (
    A2AAdapterError,
    A2AConnectionError,
    A2ATimeoutError,
)
from .types import A2AResult, A2AStreamEvent

logger = logging.getLogger(__name__)

# A2A protocol kwargs accepted by send_message / stream_message.
# Everything else (e.g. LangChain metadata, tags) is filtered out.
_A2A_SEND_KWARGS: frozenset[str] = frozenset({"context_id", "task_id", "metadata"})


class A2ARunnable(Runnable[A2AInput, A2AResult]):
    """LangChain Runnable wrapping an A2A agent.

    This is the primary integration point. It preserves the full A2A
    protocol semantics: multi-turn conversations, streaming, task
    lifecycle, and structured artifacts.

    Accepts both text and structured dict input:
    - ``str`` → sent as a ``TextPart``
    - ``dict`` → sent as a ``DataPart`` (structured JSON)

    Usage:
        # Direct invocation (text)
        a2a = await A2ARunnable.from_agent_url("http://agent:8080")
        result = await a2a.ainvoke("summarize this document")

        # Structured data input
        result = await a2a.ainvoke({"action": "analyze", "target": "sales_q4"})
        print(result.data)  # structured response payloads

        # Streaming
        async for event in a2a.astream("explain in detail"):
            print(event.text, end="")

        # Multi-turn
        a2a_conv = a2a.with_context(result.context_id)
        followup = await a2a_conv.ainvoke("elaborate on point 3")

        # As a tool for LLM function calling
        tool = a2a.as_tool()
        llm_with_tools = llm.bind_tools([tool])
    """

    def __init__(
        self,
        client: A2AClientWrapper,
        *,
        context_id: str | None = None,
    ) -> None:
        self._client = client
        self._context_id = context_id

    @classmethod
    async def from_agent_url(
        cls,
        url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        auth: A2AAuthConfig | None = None,
        transport: str | None = None,
    ) -> A2ARunnable:
        """Create an A2ARunnable by discovering the agent card at the URL.

        Args:
            url: Agent endpoint URL.
            timeout: Request timeout in seconds.
            headers: Additional HTTP headers.
            auth: Authentication configuration (mTLS, Bearer, API key, etc.).
            transport: Transport type - None for auto-detect, 'http' for HTTP,
                       'grpc' for gRPC (requires [grpc] extra).

        Returns:
            Configured A2ARunnable instance.

        Raises:
            A2AConnectionError: If agent discovery fails.
            ImportError: If gRPC transport requested but not installed.
        """
        client = A2AClientWrapper(
            url,
            timeout=timeout,
            headers=headers,
            auth=auth,
            transport=transport,
        )
        await client.get_agent_card()  # trigger discovery

        # Log if agent requires mTLS
        if client.requires_mTLS():
            logger.info("Agent '%s' requires mTLS", url)

        # Log transport selection
        actual_transport = transport or "auto-detected"
        if (
            not transport
            and client.agent_card
            and hasattr(client.agent_card, "preferred_transport")
        ):
            actual_transport = client.agent_card.preferred_transport or "auto-detected"
        logger.info(
            "Initialized A2A agent '%s' with %s transport",
            client.agent_card.name if client.agent_card else "unknown",
            actual_transport,
        )

        return cls(client)

    @property
    def agent_card(self) -> Any:
        return self._client.agent_card

    @property
    def protocol_version(self) -> str | None:
        """Get agent's A2A protocol version."""
        card = self._client.agent_card
        if card and hasattr(card, "protocol_version"):
            return card.protocol_version
        return None

    @property
    def documentation_url(self) -> str | None:
        """Get agent's documentation URL."""
        card = self._client.agent_card
        if card and hasattr(card, "documentation_url"):
            return card.documentation_url
        return None

    @property
    def preferred_transport(self) -> str | None:
        """Get agent's preferred transport (http, grpc, rest)."""
        card = self._client.agent_card
        if card and hasattr(card, "preferred_transport"):
            return card.preferred_transport
        return None

    def with_context(self, context_id: str) -> A2ARunnable:
        """Return a new A2ARunnable bound to a conversation context.

        Enables multi-turn conversations: the returned runnable will
        send all subsequent messages within the same A2A contextId.
        """
        return A2ARunnable(self._client, context_id=context_id)

    async def ainvoke(
        self,
        input: A2AInput,
        config: RunnableConfig | None = None,
        *,
        files: list[tuple[str, bytes, str]] | None = None,
        **kwargs: Any,
    ) -> A2AResult:
        """Send a message and return the full A2AResult.

        Args:
            input: Text string or structured dict payload.
            config: LangChain runnable config with callbacks, tags, metadata.
            files: Optional list of (filename, file_bytes, mime_type) tuples.

        Returns:
            A2AResult with task state, text, data, and artifacts.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2ATimeoutError: If request times out.
            A2AProtocolError: If agent returns JSON-RPC error.

        Maps to A2A ``message/send`` (blocking mode).
        """
        config = ensure_config(config)

        # Separate A2A protocol kwargs from LangChain kwargs
        a2a_kwargs = {k: v for k, v in kwargs.items() if k in _A2A_SEND_KWARGS}
        unsupported = set(kwargs) - _A2A_SEND_KWARGS
        if unsupported:
            logger.debug(
                "Ignoring non-A2A kwargs in ainvoke: %s "
                "(use 'config' for LangChain metadata/tags)",
                unsupported,
            )

        result = await self._client.send_message(
            input, files=files, context_id=self._context_id, **a2a_kwargs
        )
        return result

    def _get_name(self) -> str:
        """Return a descriptive name for tracing."""
        card = self._client.agent_card
        return f"A2ARunnable({card.name})" if card else "A2ARunnable"

    def invoke(
        self,
        input: A2AInput,
        config: RunnableConfig | None = None,
        *,
        files: list[tuple[str, bytes, str]] | None = None,
        **kwargs: Any,
    ) -> A2AResult:
        """Synchronous invocation (safe in all contexts).

        Uses LangChain's run_in_executor to safely run the async
        ainvoke() in Jupyter, FastAPI, LangServe, and other event
        loop contexts.

        Args:
            input: Text string or structured dict payload.
            config: LangChain runnable config.
            files: Optional list of (filename, file_bytes, mime_type) tuples.

        Returns:
            A2AResult with task state, text, data, and artifacts.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2ATimeoutError: If request times out.
            A2AProtocolError: If agent returns JSON-RPC error.
        """
        # Filter kwargs before forwarding to ainvoke
        a2a_kwargs = {k: v for k, v in kwargs.items() if k in _A2A_SEND_KWARGS}
        unsupported = set(kwargs) - _A2A_SEND_KWARGS
        if unsupported:
            logger.debug(
                "Ignoring non-A2A kwargs in invoke: %s "
                "(use 'config' for LangChain metadata/tags)",
                unsupported,
            )

        return run_in_executor(  # type: ignore[return-value]
            config,
            self.ainvoke,
            input,
            config,
            files=files,
            **a2a_kwargs,
        )

    async def astream(  # type: ignore[override]
        self,
        input: A2AInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[A2AStreamEvent]:
        """Stream a message response via SSE.

        Maps to A2A ``message/stream``. Yields A2AStreamEvent objects
        with extracted text, structured data, and status information.
        """
        # Filter to A2A-supported kwargs only
        a2a_kwargs = {k: v for k, v in kwargs.items() if k in _A2A_SEND_KWARGS}
        unsupported = set(kwargs) - _A2A_SEND_KWARGS
        if unsupported:
            logger.debug(
                "Ignoring non-A2A kwargs in astream: %s "
                "(use 'config' for LangChain metadata/tags)",
                unsupported,
            )

        async for event in self._client.stream_message(
            input, context_id=self._context_id, **a2a_kwargs
        ):
            yield event

    async def aresubscribe(
        self,
        task_id: str,
    ) -> AsyncIterator[A2AStreamEvent]:
        """Reconnect to an interrupted streaming task.

        This is useful for resuming long-running tasks after network
        interruption or client restart. The task must still be active
        on the agent side.

        Args:
            task_id: ID of task to resubscribe to.

        Yields:
            A2AStreamEvent instances as task continues.

        Raises:
            A2ATaskNotFoundError: If task not found or expired.
            A2AUnsupportedOperationError: If task is not streamable.

        Example:
            ```python
            # Initial streaming interrupted after first event
            async for event in agent.astream("Long task"):
                task_id = event.task_id
                break  # Simulated interruption

            # Later, reconnect to same task
            async for event in agent.aresubscribe(task_id):
                print(f"Resumed: {event.text}")
            ```
        """
        async for event in self._client.resubscribe_task(
            task_id=task_id,
            context_id=self._context_id,
        ):
            yield event

    async def get_task(self, task_id: str) -> A2AResult:
        """Poll a task's current state via tasks/get."""
        return await self._client.get_task(task_id)

    async def cancel_task(self, task_id: str) -> A2AResult:
        """Cancel a running task via tasks/cancel."""
        return await self._client.cancel_task(task_id)

    def with_default_retry(
        self,
        *,
        max_attempts: int = 3,
    ) -> Runnable[A2AInput, A2AResult]:
        """Return a retry-wrapped runnable with sensible defaults.

        Retries on transient failures (connection, timeout) with exponential
        backoff and jitter. Does not retry on auth, capability, protocol,
        or other non-transient errors.

        Args:
            max_attempts: Maximum number of retry attempts (default 3).

        Returns:
            A Runnable that automatically retries on transient failures.

        Example:
            ```python
            agent = await A2ARunnable.from_agent_url("http://agent:8080")
            resilient = agent.with_default_retry(max_attempts=5)
            result = await resilient.ainvoke("important request")
            ```
        """
        return self.with_retry(
            retry_if_exception_type=(A2AConnectionError, A2ATimeoutError),
            stop_after_attempt=max_attempts,
            wait_exponential_jitter=True,
        )

    def _make_tool(
        self,
        *,
        tool_name: str,
        tool_description: str,
        metadata: dict[str, Any] | None = None,
    ) -> BaseTool:
        """Create a single BaseTool with the given name/description.

        Args:
            tool_name: Name for the tool.
            tool_description: Description for the tool.
            metadata: Optional message-level metadata (e.g., {"skill": "skill-id"}).
        """
        from pydantic import BaseModel, Field

        runnable = self

        class QueryInput(BaseModel):
            """Input schema for A2A tool."""

            query: str = Field(
                ..., description="The query or message to send to the agent"
            )
            data: dict[str, Any] | None = Field(
                None, description="Structured data payload (alternative to query)"
            )

        class _A2ATool(BaseTool):
            name: str = tool_name
            description: str = tool_description
            args_schema: type[BaseModel] = QueryInput

            async def _arun(
                self, query: str, data: dict[str, Any] | None = None, **kwargs: Any
            ) -> str:
                try:
                    # Determine if agent expects DataPart input
                    card = runnable._client.agent_card
                    modes = card.default_input_modes if card else None
                    expects_data = modes is not None and "application/json" in modes

                    if expects_data:
                        # Always send as DataPart — merge query into data dict
                        input_data: A2AInput = {**(data or {}), "query": query}
                    elif data:
                        input_data = data
                    else:
                        input_data = query
                    result = await runnable.ainvoke(input_data, metadata=metadata)
                except A2AAdapterError as e:
                    # Map adapter exceptions to ToolException
                    raise ToolException(f"A2A tool '{self.name}' failed: {e}") from e

                # Handle non-terminal states
                if result.requires_input:
                    return (
                        f"[Agent '{self.name}' requires additional input] "
                        f"{result.text or 'Please provide more details.'}"
                    )

                if result.status == "auth-required":
                    raise ToolException(
                        f"A2A agent '{self.name}' requires authentication. "
                        "Configure auth via A2AAuthConfig."
                    )

                if result.status in ("canceled", "rejected", "unknown"):
                    raise ToolException(
                        f"A2A agent '{self.name}' returned non-success state: "
                        f"{result.status}"
                    )

                # Return JSON with text + file info if files present
                if result.files:
                    response = {
                        "text": result.text or "",
                        "files": [
                            {
                                "name": f.get("name"),
                                "mime_type": f.get("mime_type"),
                                "has_bytes": "bytes" in f,
                                "has_uri": "uri" in f,
                            }
                            for f in result.files
                        ],
                    }
                    return json.dumps(response)

                # Prefer structured data (DataPart) over text for tool outputs
                # This ensures search results etc. are returned even when
                # a short streaming summary is also present
                if result.data:
                    # If both text and data, include text as context
                    if result.text:
                        return json.dumps({"answer": result.text, "data": result.data})
                    return json.dumps(result.data)
                if result.text:
                    return result.text
                return ""

            def _run(
                self, query: str, data: dict[str, Any] | None = None, **kwargs: Any
            ) -> str:
                import asyncio
                import concurrent.futures

                async def run_async() -> str:
                    return await self._arun(query, data=data, **kwargs)

                try:
                    # If in a running loop, use a thread executor
                    asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_async())
                        return future.result()
                except RuntimeError:
                    # No running loop, safe to use asyncio.run() directly
                    return asyncio.run(run_async())

        return _A2ATool()

    @staticmethod
    def _sanitize_name(value: str) -> str:
        return value.lower().replace(" ", "_").replace("-", "_")

    def as_tools(self) -> list[BaseTool]:
        """Create one LangChain BaseTool per agent skill.

        Each skill defined in the agent's AgentCard becomes its own tool,
        giving the LLM granular control over which capability to invoke.
        All tools share the same underlying A2ARunnable — the remote agent
        decides how to route based on the message content.

        Enhanced with skill examples, tags, and input/output modes if available.

        Falls back to a single agent-level tool if the agent has no skills.
        """
        card = self._client.agent_card
        if not card or not card.skills:
            return [self.as_tool()]

        tools: list[BaseTool] = []
        for skill in card.skills:
            tool_name = self._sanitize_name(skill.id or skill.name)
            tool_desc = skill.description or f"Skill: {skill.name}"

            # Add examples if present (a2a-sdk 0.3.19+ feature)
            if hasattr(skill, "examples") and skill.examples:
                examples_text = "\n\nExamples:\n"
                for ex in skill.examples[:3]:  # Limit to 3 examples
                    examples_text += f"- {ex}\n"
                tool_desc += examples_text

            # Add tags if present
            if hasattr(skill, "tags") and skill.tags:
                tags_text = f"\n\nTags: {', '.join(skill.tags)}"
                tool_desc += tags_text

            # Add input/output modes if present
            if hasattr(skill, "input_modes") and skill.input_modes:
                modes_str = ", ".join(skill.input_modes)
                tool_desc += f"\n\nAccepted input types: {modes_str}"

            if hasattr(skill, "output_modes") and skill.output_modes:
                modes_str = ", ".join(skill.output_modes)
                tool_desc += f"\n\nOutput types: {modes_str}"

            # Build skill routing metadata
            skill_metadata = {"skill": skill.id} if skill.id else None

            tools.append(
                self._make_tool(
                    tool_name=tool_name,
                    tool_description=tool_desc,
                    metadata=skill_metadata,
                )
            )

        return tools

    def as_tool(  # type: ignore[override]
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> BaseTool:
        """Create a single LangChain BaseTool for the entire agent.

        Use ``as_tools()`` instead to get one tool per skill.
        This method is useful when you want to expose the agent as
        a single coarse-grained tool.

        Enhanced with protocol version and documentation URL if available.
        """
        card = self._client.agent_card
        tool_name = name or (self._sanitize_name(card.name) if card else "a2a_agent")

        # Use custom description if provided, otherwise use card description
        if description:
            tool_desc = description
        else:
            tool_desc = (
                card.description if card else None
            ) or "Interact with A2A agent"

            # Add protocol version and documentation URL if using default desc
            if card:
                if hasattr(card, "protocol_version") and card.protocol_version:
                    version = card.protocol_version
                    tool_desc += f"\n\nProtocol version: {version}"

                if hasattr(card, "documentation_url") and card.documentation_url:
                    docs_url = card.documentation_url
                    tool_desc += f"\n\nDocs: {docs_url}"

        return self._make_tool(tool_name=tool_name, tool_description=tool_desc)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> A2ARunnable:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
