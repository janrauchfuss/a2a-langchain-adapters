"""Tests for task resubscribe functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from a2a.types import (
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from a2a_langchain_adapters import A2ARunnable, A2AStreamEvent
from a2a_langchain_adapters.client_wrapper import A2AClientWrapper
from a2a_langchain_adapters.exceptions import A2ATaskNotFoundError


class TestResubscribeTask:
    """Test task resubscribe at client wrapper level."""

    @pytest.mark.asyncio
    async def test_resubscribe_success(self, client_wrapper):
        """Resubscribe to active streaming task."""

        # Mock SDK resubscribe
        async def mock_resubscribe(request):
            task_id = request.params.id
            yield TaskStatusUpdateEvent(
                task_id=task_id,
                context_id="c1",
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id="c1",
                artifact=Artifact(
                    artifact_id="a1",
                    parts=[Part(root=TextPart(text="Resumed"))],
                ),
            )

        client_wrapper._a2a_client.resubscribe = mock_resubscribe

        events = []
        async for event in client_wrapper.resubscribe_task("task-123"):
            events.append(event)

        assert len(events) == 2
        assert events[0].kind == "status-update"
        assert events[1].kind == "artifact-update"
        assert events[1].text == "Resumed"

    @pytest.mark.asyncio
    async def test_resubscribe_task_not_found(self, client_wrapper):
        """Resubscribe raises if task expired."""

        async def mock_resubscribe(request):
            if True:  # Force exception on first iteration
                raise Exception("Task not found")
            yield  # Make it an async generator

        client_wrapper._a2a_client.resubscribe = mock_resubscribe

        with pytest.raises(A2ATaskNotFoundError, match="not found or expired"):
            async for _ in client_wrapper.resubscribe_task("old-task"):
                pass


class TestRunnableResubscribe:
    """Test aresubscribe() on A2ARunnable."""

    @pytest.mark.asyncio
    async def test_aresubscribe(self):
        """aresubscribe() passes through to client."""
        mock_wrapper = AsyncMock(spec=A2AClientWrapper)

        async def mock_resubscribe(task_id, context_id):
            yield A2AStreamEvent(
                kind="artifact-update",
                task_id=task_id,
                context_id="c1",
                text="Resumed",
            )

        mock_wrapper.resubscribe_task = mock_resubscribe

        runnable = A2ARunnable(mock_wrapper)

        events = []
        async for event in runnable.aresubscribe("task-123"):
            events.append(event)

        assert len(events) == 1
        assert events[0].text == "Resumed"

    @pytest.mark.asyncio
    async def test_aresubscribe_with_context_id(self):
        """aresubscribe() passes context_id from runnable."""
        mock_wrapper = AsyncMock(spec=A2AClientWrapper)

        captured_args = {}

        async def mock_resubscribe(task_id, context_id):
            captured_args["task_id"] = task_id
            captured_args["context_id"] = context_id
            yield A2AStreamEvent(
                kind="artifact-update",
                task_id=task_id,
                context_id=context_id,
            )

        mock_wrapper.resubscribe_task = mock_resubscribe

        # Create runnable with a specific context
        runnable = A2ARunnable(mock_wrapper, context_id="ctx-123")

        async for _ in runnable.aresubscribe("task-456"):
            pass

        assert captured_args["task_id"] == "task-456"
        assert captured_args["context_id"] == "ctx-123"
