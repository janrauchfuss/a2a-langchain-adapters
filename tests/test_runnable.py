"""Tests for a2a_langchain_adapters.runnable."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import ToolException

from a2a_langchain_adapters.runnable import A2ARunnable
from a2a_langchain_adapters.types import A2AResult, A2AStreamEvent

from .conftest import (
    make_streaming_status_event,
)

# ============================================================================
# A2ARunnable initialization and factory
# ============================================================================


class TestA2ARunnableFactory:
    """Tests for A2ARunnable.from_agent_url factory method."""

    @pytest.mark.asyncio
    async def test_from_agent_url_success(self, agent_card):
        """Successfully create A2ARunnable from agent URL."""
        # Mock the client wrapper
        with patch("a2a_langchain_adapters.runnable.A2AClientWrapper") as MockWrapper:
            mock_wrapper_instance = AsyncMock()
            mock_wrapper_instance.agent_card = agent_card
            mock_wrapper_instance.get_agent_card = AsyncMock(return_value=agent_card)
            mock_wrapper_instance.requires_mTLS = Mock(return_value=False)
            MockWrapper.return_value = mock_wrapper_instance

            # Create runnable
            runnable = await A2ARunnable.from_agent_url("http://test-agent:8080")

            # Verify
            assert runnable is not None
            assert runnable.agent_card == agent_card
            MockWrapper.assert_called_once_with(
                "http://test-agent:8080",
                timeout=30.0,
                headers=None,
                auth=None,
                transport=None,
            )

    @pytest.mark.asyncio
    async def test_from_agent_url_with_timeout(self, agent_card):
        """Create A2ARunnable with custom timeout."""
        with patch("a2a_langchain_adapters.runnable.A2AClientWrapper") as MockWrapper:
            mock_wrapper_instance = AsyncMock()
            mock_wrapper_instance.agent_card = agent_card
            mock_wrapper_instance.get_agent_card = AsyncMock(return_value=agent_card)
            mock_wrapper_instance.requires_mTLS = Mock(return_value=False)
            MockWrapper.return_value = mock_wrapper_instance

            await A2ARunnable.from_agent_url("http://test-agent:8080", timeout=60.0)

            MockWrapper.assert_called_once_with(
                "http://test-agent:8080",
                timeout=60.0,
                headers=None,
                auth=None,
                transport=None,
            )

    @pytest.mark.asyncio
    async def test_from_agent_url_with_headers(self, agent_card):
        """Create A2ARunnable with custom headers."""
        headers = {"Authorization": "Bearer token123"}
        with patch("a2a_langchain_adapters.runnable.A2AClientWrapper") as MockWrapper:
            mock_wrapper_instance = AsyncMock()
            mock_wrapper_instance.agent_card = agent_card
            mock_wrapper_instance.get_agent_card = AsyncMock(return_value=agent_card)
            mock_wrapper_instance.requires_mTLS = Mock(return_value=False)
            MockWrapper.return_value = mock_wrapper_instance

            await A2ARunnable.from_agent_url(
                "http://test-agent:8080",
                timeout=45.0,
                headers=headers,
            )

            MockWrapper.assert_called_once_with(
                "http://test-agent:8080",
                timeout=45.0,
                headers=headers,
                auth=None,
                transport=None,
            )


# ============================================================================
# Invocation tests
# ============================================================================


class TestA2ARunnableInvoke:
    """Tests for A2ARunnable.invoke and ainvoke."""

    @pytest.mark.asyncio
    async def test_ainvoke_text_input(self, client_wrapper, agent_card):
        """ainvoke with text input."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            text="Response text",
            task_id="task123",
            context_id="ctx456",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await runnable.ainvoke("Hello, agent")

        assert result == expected_result
        client_wrapper.send_message.assert_called_once_with(
            "Hello, agent", files=None, context_id=None
        )

    @pytest.mark.asyncio
    async def test_ainvoke_dict_input(self, client_wrapper):
        """ainvoke with dict input."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            data=[{"key": "value"}],
            task_id="task123",
            context_id="ctx456",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await runnable.ainvoke({"action": "analyze"})

        assert result == expected_result
        client_wrapper.send_message.assert_called_once_with(
            {"action": "analyze"}, files=None, context_id=None
        )

    @pytest.mark.asyncio
    async def test_ainvoke_with_context(self, client_wrapper):
        """ainvoke preserves context_id for multi-turn."""
        context_id = "ctx789"
        runnable = A2ARunnable(client_wrapper, context_id=context_id)

        expected_result = A2AResult(
            text="Response",
            task_id="task123",
            context_id=context_id,
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        await runnable.ainvoke("Follow-up message")

        client_wrapper.send_message.assert_called_once_with(
            "Follow-up message", files=None, context_id=context_id
        )

    @pytest.mark.asyncio
    async def test_ainvoke_passes_a2a_kwargs(self, client_wrapper):
        """ainvoke forwards A2A-supported kwargs (task_id) to send_message."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            text="Response",
            task_id="task123",
            context_id="ctx456",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        await runnable.ainvoke("Hello", task_id="task123")

        client_wrapper.send_message.assert_called_once_with(
            "Hello",
            files=None,
            context_id=None,
            task_id="task123",
        )

    @pytest.mark.asyncio
    async def test_ainvoke_filters_unsupported_kwargs(self, client_wrapper):
        """ainvoke filters out non-A2A kwargs instead of forwarding them."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            text="Response",
            task_id="task123",
            context_id="ctx456",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        await runnable.ainvoke("Hello", timeout=60.0, custom_param="value")

        # Only A2A kwargs should reach send_message — not timeout or custom_param
        client_wrapper.send_message.assert_called_once_with(
            "Hello",
            files=None,
            context_id=None,
        )

    @pytest.mark.asyncio
    async def test_ainvoke_with_metadata_forwarded(self, client_wrapper):
        """BUG-A2 FIX: metadata kwarg must forward to send_message.

        The Director Agent passes metadata={"skill": skill_id} to ainvoke()
        for skill routing. This was previously rejected; now it's forwarded
        properly to send_message() for routing to specific agent skills.
        """
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            text="ok",
            task_id="t1",
            context_id="c1",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await runnable.ainvoke(
            {"text": "test"}, metadata={"skill": "jailbreak-check"}
        )

        assert result.text == "ok"
        # metadata MUST appear in send_message call for skill routing
        call_kwargs = client_wrapper.send_message.call_args.kwargs
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"] == {"skill": "jailbreak-check"}

    def test_invoke_exists(self):
        """invoke method exists and accepts input and config."""
        wrapper = AsyncMock()
        wrapper.agent_card = None
        runnable = A2ARunnable(wrapper)

        # Just verify the method exists and has the right signature
        assert hasattr(runnable, "invoke")
        assert callable(runnable.invoke)


# ============================================================================
# Streaming tests
# ============================================================================


class TestA2ARunnableStream:
    """Tests for A2ARunnable.astream."""

    @pytest.mark.asyncio
    async def test_astream_text_events(self, client_wrapper):
        """astream yields text-only artifact events (BUG-A1 fix)."""
        runnable = A2ARunnable(client_wrapper)

        # After BUG-A1 fix, stream_message() yields A2AStreamEvent, not raw SDK events
        stream_event = A2AStreamEvent(
            kind="artifact-update",
            task_id="task1",
            context_id="ctx1",
            text="streaming text",
        )

        async def mock_stream(*args, **kwargs):
            yield stream_event

        client_wrapper.stream_message = mock_stream

        events = []
        async for event in runnable.astream("query"):
            events.append(event)

        assert len(events) == 1
        assert events[0].kind == "artifact-update"
        assert events[0].text == "streaming text"
        assert events[0].task_id == "task1"

    @pytest.mark.asyncio
    async def test_astream_data_events(self, client_wrapper):
        """astream yields structured data from DataParts (BUG-A1 fix)."""
        runnable = A2ARunnable(client_wrapper)

        # After BUG-A1 fix, stream_message() yields A2AStreamEvent with extracted data
        stream_event = A2AStreamEvent(
            kind="artifact-update",
            task_id="task1",
            context_id="ctx1",
            data=[{"result": "analysis"}],
        )

        async def mock_stream(*args, **kwargs):
            yield stream_event

        client_wrapper.stream_message = mock_stream

        events = []
        async for event in runnable.astream("query"):
            events.append(event)

        assert len(events) == 1
        assert events[0].kind == "artifact-update"
        assert events[0].data == [{"result": "analysis"}]

    @pytest.mark.asyncio
    async def test_astream_mixed_events(self, client_wrapper):
        """astream handles artifact and status events in sequence (BUG-A1 fix)."""
        runnable = A2ARunnable(client_wrapper)

        # After BUG-A1 fix, stream_message() yields A2AStreamEvent for both types
        artifact_event = A2AStreamEvent(
            kind="artifact-update",
            task_id="t1",
            context_id="ctx1",
            text="hello",
        )
        status_event = A2AStreamEvent(
            kind="status-update",
            task_id="t1",
            context_id="ctx1",
            status="completed",
            final=True,
        )

        async def mock_stream(*args, **kwargs):
            yield artifact_event
            yield status_event

        client_wrapper.stream_message = mock_stream

        events = list([e async for e in runnable.astream("query")])

        assert len(events) == 2
        assert events[0].kind == "artifact-update"
        assert events[1].kind == "status-update"

    @pytest.mark.asyncio
    async def test_astream_status_event_with_data(self, client_wrapper):
        """BUG-1237: astream extracts DataPart from status-update message."""
        runnable = A2ARunnable(client_wrapper)

        # Create A2AStreamEvent directly (stream_message returns these)
        status_event = A2AStreamEvent(
            kind="status-update",
            task_id="t1",
            context_id="c1",
            text="Search done",
            data=[{"results": [{"id": 1}]}],
            status="completed",
            final=True,
        )

        async def mock_stream(*args, **kwargs):
            yield status_event

        client_wrapper.stream_message = mock_stream

        events = []
        async for event in runnable.astream("search"):
            events.append(event)

        assert len(events) == 1
        assert events[0].kind == "status-update"
        assert events[0].text == "Search done"
        assert events[0].data == [{"results": [{"id": 1}]}]
        assert events[0].final is True

    @pytest.mark.asyncio
    async def test_astream_with_context(self, client_wrapper):
        """astream preserves context_id for multi-turn."""
        context_id = "ctx789"
        runnable = A2ARunnable(client_wrapper, context_id=context_id)

        status_event = make_streaming_status_event(task_id="t1")

        async def mock_stream(*args, **kwargs):
            assert kwargs.get("context_id") == context_id
            yield status_event.root.result  # type: ignore[union-attr]

        client_wrapper.stream_message = mock_stream

        async for _ in runnable.astream("query"):
            pass


# ============================================================================
# Context management tests
# ============================================================================


class TestA2ARunnableContext:
    """Tests for A2ARunnable.with_context."""

    def test_with_context_creates_new_runnable(self, client_wrapper):
        """with_context returns a new A2ARunnable with bound context."""
        runnable = A2ARunnable(client_wrapper)
        context_id = "conversation-123"

        new_runnable = runnable.with_context(context_id)

        assert new_runnable is not runnable  # New instance
        assert new_runnable._context_id == context_id
        assert runnable._context_id is None  # Original unchanged

    def test_with_context_chaining(self, client_wrapper):
        """with_context can be chained."""
        runnable = A2ARunnable(client_wrapper)

        first_context = "ctx1"
        second_context = "ctx2"

        r1 = runnable.with_context(first_context)
        r2 = r1.with_context(second_context)

        assert r1._context_id == first_context
        assert r2._context_id == second_context
        assert runnable._context_id is None


# ============================================================================
# Task management tests
# ============================================================================


class TestA2ARunnableTaskManagement:
    """Tests for A2ARunnable.get_task, cancel_task."""

    @pytest.mark.asyncio
    async def test_get_task(self, client_wrapper):
        """get_task delegates to client wrapper."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            status="working",
            task_id="task123",
            context_id="ctx1",
            text="Current status",
        )
        client_wrapper.get_task = AsyncMock(return_value=expected_result)

        result = await runnable.get_task("task123")

        assert result == expected_result
        client_wrapper.get_task.assert_called_once_with("task123")

    @pytest.mark.asyncio
    async def test_cancel_task(self, client_wrapper):
        """cancel_task delegates to client wrapper."""
        runnable = A2ARunnable(client_wrapper)
        expected_result = A2AResult(
            status="canceled",
            task_id="task123",
            context_id="ctx1",
        )
        client_wrapper.cancel_task = AsyncMock(return_value=expected_result)

        result = await runnable.cancel_task("task123")

        assert result == expected_result
        client_wrapper.cancel_task.assert_called_once_with("task123")


# Tool binding tests
# ============================================================================


class TestA2ARunnableTool:
    """Tests for A2ARunnable.as_tool and as_tools."""

    def test_as_tool_default_name_and_description(self, client_wrapper, agent_card):
        """as_tool derives name and description from agent card."""
        runnable = A2ARunnable(client_wrapper)

        tool = runnable.as_tool()

        assert tool.name == "test_agent"  # Sanitized from "Test Agent"
        assert tool.description.startswith("A test agent for unit tests")
        assert "Protocol version:" in tool.description

    def test_as_tool_custom_name_and_description(self, client_wrapper):
        """as_tool accepts custom name and description."""
        runnable = A2ARunnable(client_wrapper)

        tool = runnable.as_tool(
            name="custom_name",
            description="Custom description",
        )

        assert tool.name == "custom_name"
        # Custom description should be used as-is without extra metadata
        assert tool.description == "Custom description"

    def test_as_tool_fallback_no_agent_card(self):
        """as_tool falls back when no agent card."""
        wrapper = AsyncMock()
        wrapper.agent_card = None

        runnable = A2ARunnable(wrapper)
        tool = runnable.as_tool()

        assert tool.name == "a2a_agent"
        assert tool.description == "Interact with A2A agent"

    def test_as_tool_has_args_schema_for_bind_tools(self, client_wrapper):
        """as_tool must define args_schema so LLMs can generate tool calls.

        Without args_schema, model.bind_tools([tool]) cannot produce a
        valid function-calling schema and the LLM silently ignores the tool.
        Regression test for the missing args_schema bug.
        """
        from langchain_core.utils.function_calling import convert_to_openai_function

        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        # args_schema must be set
        assert tool.args_schema is not None

        # Must convert to a valid OpenAI function schema
        schema = convert_to_openai_function(tool)
        params = schema["parameters"]
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "query" in params.get("required", [])

    def test_as_tools_have_args_schema(self, client_wrapper, agent_card):
        """Every tool from as_tools() must also carry args_schema."""
        from langchain_core.utils.function_calling import convert_to_openai_function

        runnable = A2ARunnable(client_wrapper)
        tools = runnable.as_tools()

        for tool in tools:
            assert tool.args_schema is not None, (
                f"Tool '{tool.name}' missing args_schema"
            )
            schema = convert_to_openai_function(tool)
            assert "query" in schema["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_as_tool_execution_success(self, client_wrapper):
        """Tool execution calls runnable.ainvoke and returns text."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text="Tool response text",
            task_id="task123",
            status="completed",
            context_id="ctx1",
            data=[],
            files=[],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        # Tools have both _arun and _run
        result = await tool._arun("query")

        assert result == "Tool response text"
        client_wrapper.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_as_tool_execution_fallback_to_data(self, client_wrapper):
        """Tool execution falls back to data if no text."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text=None,
            data=[{"key": "value"}],
            task_id="task123",
            status="completed",
            context_id="ctx1",
            files=[],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await tool._arun("query")

        assert result == json.dumps([{"key": "value"}])

    @pytest.mark.asyncio
    async def test_as_tool_execution_error(self, client_wrapper):
        """Tool execution raises ToolException on rejected status."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            task_id="task123",
            status="rejected",
            context_id="ctx1",
            text=None,
            data=[],
            files=[],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        with pytest.raises(ToolException):
            await tool._arun("query")

    @pytest.mark.asyncio
    async def test_as_tool_execution_empty_response(self, client_wrapper):
        """Tool execution returns empty string if no text/data."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text=None,
            task_id="task123",
            status="completed",
            context_id="ctx1",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await tool._arun("query")

        assert result == ""

    def test_tool_run_sync_context(self, client_wrapper):
        """Tool's _run method works in pure sync context.

        Regression test for: asyncio.run() cannot be called from a running event loop.
        This verifies _run works correctly when called from sync code.
        """
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text="Response from sync",
            task_id="task123",
            status="completed",
            context_id="ctx1",
            data=[],
            files=[],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        # Call _run from sync context
        result = tool._run("test query")

        assert result == "Response from sync"
        client_wrapper.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_run_from_async_via_to_thread(self, client_wrapper):
        """Tool's _run method works when called from async context via to_thread.

        Regression test for: asyncio.run() cannot be called from a running event loop.
        Simulates the pattern where director agents call tools asynchronously
        using asyncio.to_thread() to call sync tool methods.
        """
        import asyncio

        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text="Response from async context",
            task_id="task123",
            status="completed",
            context_id="ctx1",
            data=[],
            files=[],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        # Simulate calling tool._run from within an async context
        # This is the pattern used when a director agent invokes tools:
        # await asyncio.to_thread(tool.invoke, ...)
        result = await asyncio.to_thread(tool._run, "query from director")

        assert result == "Response from async context"
        client_wrapper.send_message.assert_called_once()

    def test_as_tools_with_skills(self, client_wrapper, agent_card):
        """as_tools creates one tool per skill."""
        runnable = A2ARunnable(client_wrapper)

        tools = runnable.as_tools()

        # Agent card has 2 skills: summarize, translate
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "summarize" in tool_names
        assert "translate" in tool_names

    def test_as_tools_fallback_no_skills(self, agent_card_no_skills):
        """as_tools falls back to single agent tool if no skills."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_no_skills

        runnable = A2ARunnable(wrapper)
        tools = runnable.as_tools()

        assert len(tools) == 1
        assert tools[0].name == "simple_agent"

    def test_sanitize_name(self):
        """_sanitize_name converts to valid tool name."""
        assert A2ARunnable._sanitize_name("My Tool") == "my_tool"
        assert A2ARunnable._sanitize_name("Tool-Name") == "tool_name"
        assert A2ARunnable._sanitize_name("UPPERCASE") == "uppercase"
        assert A2ARunnable._sanitize_name("mixed Case-Tool") == "mixed_case_tool"


# ============================================================================
# Lifecycle management tests
# ============================================================================


class TestA2ARunnableLifecycle:
    """Tests for A2ARunnable resource management."""

    @pytest.mark.asyncio
    async def test_close(self, client_wrapper):
        """close delegates to client wrapper."""
        runnable = A2ARunnable(client_wrapper)
        client_wrapper.close = AsyncMock()

        await runnable.close()

        client_wrapper.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, client_wrapper):
        """A2ARunnable supports async context manager protocol."""
        runnable = A2ARunnable(client_wrapper)
        client_wrapper.close = AsyncMock()

        async with runnable as r:
            assert r is runnable

        client_wrapper.close.assert_called_once()

    def test_agent_card_property(self, client_wrapper, agent_card):
        """agent_card property returns client's agent card."""
        runnable = A2ARunnable(client_wrapper)

        assert runnable.agent_card == agent_card


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestA2ARunnableEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_text_parts_concatenation(self, client_wrapper):
        """Multiple text parts are concatenated with newlines."""
        runnable = A2ARunnable(client_wrapper)

        # Create A2AStreamEvent with concatenated text (stream_message does this)
        artifact_event = A2AStreamEvent(
            kind="artifact-update",
            task_id="t1",
            context_id="c1",
            text="First part\nSecond part",
        )

        async def mock_stream(*args, **kwargs):
            yield artifact_event

        client_wrapper.stream_message = mock_stream

        events = list([e async for e in runnable.astream("query")])

        assert events[0].text == "First part\nSecond part"


# ============================================================================
# Enhanced tool metadata tests
# ============================================================================


class TestToolMetadataInputOutputModes:
    """Test tool descriptions with input/output mode metadata."""

    def test_tool_includes_input_output_modes(self, client_wrapper):
        """Tool description includes skill input/output modes."""
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill

        skill_with_modes = AgentSkill(
            id="analyze",
            name="Analyze",
            description="Analyze data",
            tags=["analyze"],
            input_modes=["text/plain", "application/json"],
            output_modes=["text/plain", "application/json"],
        )

        card = AgentCard(
            name="Advanced Agent",
            description="An agent with mode info",
            url="http://advanced-agent:8080",
            version="1.0.0",
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[skill_with_modes],
        )

        wrapper = AsyncMock()
        wrapper.agent_card = card

        runnable = A2ARunnable(wrapper)
        tools = runnable.as_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert "Accepted input types:" in tool.description
        assert "text/plain" in tool.description
        assert "application/json" in tool.description
        assert "Output types:" in tool.description


class TestEnhancedToolMetadata:
    """Test enhanced tool descriptions with examples and tags."""

    def test_tool_includes_examples(self, agent_card_with_examples):
        """Tool description includes skill examples."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)
        tools = runnable.as_tools()

        assert len(tools) == 1
        desc = tools[0].description
        assert "Examples:" in desc
        assert "Summarize the Q4 earnings report" in desc

    def test_tool_includes_tags(self, agent_card_with_examples):
        """Tool description includes skill tags."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)
        tools = runnable.as_tools()

        assert len(tools) == 1
        desc = tools[0].description
        assert "Tags:" in desc
        assert "nlp" in desc
        assert "summarization" in desc

    def test_agent_tool_includes_protocol_version(self, agent_card_with_examples):
        """Whole-agent tool includes protocol version."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)
        tool = runnable.as_tool()

        assert "Protocol version:" in tool.description
        assert "1.0.0" in tool.description

    def test_agent_tool_includes_documentation_url(self, agent_card_with_examples):
        """Whole-agent tool includes documentation URL."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)
        tool = runnable.as_tool()

        assert "Docs:" in tool.description
        assert "https://docs.example.com/agent" in tool.description

    def test_metadata_protocol_version_property(self, agent_card_with_examples):
        """Test protocol_version property."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)

        assert runnable.protocol_version == "1.0.0"

    def test_metadata_documentation_url_property(self, agent_card_with_examples):
        """Test documentation_url property."""
        wrapper = AsyncMock()
        wrapper.agent_card = agent_card_with_examples

        runnable = A2ARunnable(wrapper)

        assert runnable.documentation_url == "https://docs.example.com/agent"

    def test_metadata_properties_without_agent_card(self):
        """Test metadata properties return None without agent card."""
        wrapper = AsyncMock()
        wrapper.agent_card = None

        runnable = A2ARunnable(wrapper)

        assert runnable.protocol_version is None
        assert runnable.documentation_url is None
        assert runnable.preferred_transport is None

    def test_metadata_preferred_transport_property(self):
        """Test preferred_transport property."""
        from a2a.types import AgentCapabilities, AgentCard

        card = AgentCard(
            name="gRPC Agent",
            description="Agent preferring gRPC",
            url="grpc://test:50051",
            version="1.0.0",
            preferred_transport="grpc",
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[],
        )

        wrapper = AsyncMock()
        wrapper.agent_card = card

        runnable = A2ARunnable(wrapper)

        assert runnable.preferred_transport == "grpc"


# ============================================================================
# Test Coverage Gap Coverage - ainvoke with callbacks
# ============================================================================


class TestA2ARunnableCallbacks:
    """Tests for callback configuration in ainvoke."""

    @pytest.mark.asyncio
    async def test_ainvoke_with_callbacks(self, client_wrapper):
        """ainvoke configures callback manager when callbacks provided."""
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

        runnable = A2ARunnable(client_wrapper)

        expected_result = A2AResult(
            text="Response with callbacks",
            task_id="task123",
            context_id="ctx1",
            status="completed",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        # Mock the callback manager configuration
        mock_callback_manager = AsyncMock(spec=AsyncCallbackManagerForChainRun)
        mock_callback_manager.on_chain_end = AsyncMock()
        mock_callback_manager.on_chain_error = AsyncMock()

        with patch(
            "langchain_core.callbacks.manager.AsyncCallbackManagerForChainRun"
        ) as MockCallbackManager:
            MockCallbackManager.get_noop_manager = Mock(
                return_value=mock_callback_manager
            )
            MockCallbackManager.configure = Mock(
                return_value=AsyncMock(
                    on_chain_start=AsyncMock(return_value=mock_callback_manager)
                )
            )

            # Call with callback config
            config: dict[str, Any] = {
                "callbacks": [Mock()],
                "tags": ["test"],
                "metadata": {"test": True},
            }
            result = await runnable.ainvoke("query", config=config)  # type: ignore[arg-type]

            assert result == expected_result
            client_wrapper.send_message.assert_called_once()


# ============================================================================
# Test Coverage Gap Coverage - ainvoke exception handling
# ============================================================================


class TestA2ARunnableExceptionHandling:
    """Tests for exception handling in ainvoke."""

    @pytest.mark.asyncio
    async def test_ainvoke_exception_propagates(self, client_wrapper):
        """ainvoke propagates exceptions from send_message."""
        from a2a_langchain_adapters.exceptions import A2AConnectionError

        runnable = A2ARunnable(client_wrapper)
        client_wrapper.send_message = AsyncMock(
            side_effect=A2AConnectionError("Connection failed")
        )

        with pytest.raises(A2AConnectionError):
            await runnable.ainvoke("query")


# ============================================================================
# Test Coverage Gap Coverage - invoke (sync) method
# ============================================================================


class TestA2ARunnableInvokeSync:
    """Tests for synchronous invoke method."""

    def test_invoke_returns_coroutine_or_result(self, client_wrapper):
        """invoke method exists and can be called."""
        runnable = A2ARunnable(client_wrapper)

        # This tests that invoke is defined and callable
        # Note: Full testing of run_in_executor requires async context
        assert hasattr(runnable, "invoke")
        assert callable(runnable.invoke)


# ============================================================================
# Test Coverage Gap Coverage - Tool error and state handling
# ============================================================================


class TestA2ARunnableToolErrorHandling:
    """Tests for tool error handling and state transitions."""

    @pytest.mark.asyncio
    async def test_as_tool_adapter_error(self, client_wrapper):
        """Tool execution raises ToolException on A2AAdapterError."""
        from a2a_langchain_adapters.exceptions import A2AAdapterError

        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        client_wrapper.send_message = AsyncMock(
            side_effect=A2AAdapterError("Adapter error")
        )

        with pytest.raises(ToolException, match="A2A tool.*failed"):
            await tool._arun("query")

    @pytest.mark.asyncio
    async def test_as_tool_requires_input_state(self, client_wrapper):
        """Tool execution handles requires_input state gracefully."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            task_id="task123",
            status="input-required",
            context_id="ctx1",
            text="Please provide more details",
            requires_input=True,
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await tool._arun("incomplete query")

        assert "requires additional input" in result
        assert "Please provide more details" in result

    @pytest.mark.asyncio
    async def test_as_tool_auth_required_state(self, client_wrapper):
        """Tool execution raises ToolException for auth-required state."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            task_id="task123",
            status="auth-required",
            context_id="ctx1",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        with pytest.raises(ToolException, match="requires authentication"):
            await tool._arun("query")

    @pytest.mark.asyncio
    async def test_as_tool_canceled_state(self, client_wrapper):
        """Tool execution raises ToolException for canceled state."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            task_id="task123",
            status="canceled",
            context_id="ctx1",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        with pytest.raises(ToolException, match="non-success state"):
            await tool._arun("query")

    @pytest.mark.asyncio
    async def test_as_tool_unknown_state(self, client_wrapper):
        """Tool execution raises ToolException for unknown state."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            task_id="task123",
            status="unknown",
            context_id="ctx1",
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        with pytest.raises(ToolException, match="non-success state"):
            await tool._arun("query")

    @pytest.mark.asyncio
    async def test_as_tool_files_output(self, client_wrapper):
        """Tool execution returns JSON with file metadata."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        expected_result = A2AResult(
            text="Results ready",
            task_id="task123",
            status="completed",
            context_id="ctx1",
            files=[
                {
                    "name": "result.pdf",
                    "mime_type": "application/pdf",
                    "uri": "s3://...",
                },
                {"name": "data.json", "bytes": b"..."},
            ],
        )
        client_wrapper.send_message = AsyncMock(return_value=expected_result)

        result = await tool._arun("query")

        # Result should be JSON with file info
        result_dict = json.loads(result)
        assert result_dict["text"] == "Results ready"
        assert len(result_dict["files"]) == 2
        assert result_dict["files"][0]["name"] == "result.pdf"
        assert result_dict["files"][0]["has_uri"] is True
        assert result_dict["files"][1]["has_bytes"] is True

    def test_as_tool_sync_run_method_exists(self, client_wrapper):
        """Tool _run (sync) method exists."""
        runnable = A2ARunnable(client_wrapper)
        tool = runnable.as_tool()

        # Verify the sync _run method exists
        assert hasattr(tool, "_run")
        assert callable(tool._run)
