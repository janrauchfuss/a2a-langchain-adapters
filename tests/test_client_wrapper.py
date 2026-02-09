"""Tests for langchain_a2a_adapters.client_wrapper."""

from __future__ import annotations

import pytest
from a2a.types import (
    DataPart,
    TaskState,
    TextPart,
)

from langchain_a2a_adapters.client_wrapper import (
    A2AClientWrapper,
    _build_message,
    _extract_parts,
    _extract_text_from_parts,
    _serialize_part,
)

from .conftest import (
    make_artifact,
    make_cancel_task_error,
    make_cancel_task_success,
    make_data_part,
    make_file_part_bytes,
    make_file_part_uri,
    make_get_task_error,
    make_get_task_success,
    make_message,
    make_send_error,
    make_send_success,
    make_streaming_artifact_event,
    make_streaming_error,
    make_streaming_status_event,
    make_task,
    make_text_part,
)

# ============================================================================
# Helper function tests
# ============================================================================


class TestExtractParts:
    def test_text_only(self):
        parts = [make_text_part("hello"), make_text_part("world")]
        texts, data, files = _extract_parts(parts)
        assert texts == ["hello", "world"]
        assert data == []
        assert files == []

    def test_data_only(self):
        parts = [make_data_part({"key": "val"}), make_data_part({"n": 42})]
        texts, data, files = _extract_parts(parts)
        assert texts == []
        assert data == [{"key": "val"}, {"n": 42}]
        assert files == []

    def test_file_uri(self):
        parts = [make_file_part_uri("https://example.com/f.pdf")]
        texts, data, files = _extract_parts(parts)
        assert texts == []
        assert data == []
        assert len(files) == 1
        assert files[0]["uri"] == "https://example.com/f.pdf"
        assert files[0]["name"] == "file.pdf"
        assert files[0]["mime_type"] == "application/pdf"

    def test_file_bytes(self):
        parts = [make_file_part_bytes("SGVsbG8=", name="hello.txt")]
        texts, data, files = _extract_parts(parts)
        assert len(files) == 1
        assert files[0]["bytes"] == "SGVsbG8="
        assert files[0]["name"] == "hello.txt"

    def test_mixed_parts(self):
        parts = [
            make_text_part("hi"),
            make_data_part({"x": 1}),
            make_file_part_uri(
                "https://example.com/img.png",
                name="img.png",
                mime_type="image/png",
            ),
        ]
        texts, data, files = _extract_parts(parts)
        assert texts == ["hi"]
        assert data == [{"x": 1}]
        assert files[0]["uri"] == "https://example.com/img.png"

    def test_empty_parts(self):
        texts, data, files = _extract_parts([])
        assert texts == []
        assert data == []
        assert files == []


class TestExtractTextFromParts:
    def test_extracts_text(self):
        parts = [
            make_text_part("a"),
            make_data_part({"k": "v"}),
            make_text_part("b"),
        ]
        assert _extract_text_from_parts(parts) == "a\nb"

    def test_empty(self):
        assert _extract_text_from_parts([]) == ""


class TestBuildMessage:
    def test_str_input(self):
        msg = _build_message("hello world")
        assert msg.role == "user"
        assert len(msg.parts) == 1
        inner = msg.parts[0].root
        assert isinstance(inner, TextPart)
        assert inner.text == "hello world"
        assert msg.context_id is None
        assert msg.task_id is None

    def test_dict_input(self):
        msg = _build_message({"action": "search", "query": "test"})
        inner = msg.parts[0].root
        assert isinstance(inner, DataPart)
        assert inner.data == {"action": "search", "query": "test"}

    def test_with_context_and_task(self):
        msg = _build_message("hi", context_id="ctx-1", task_id="task-1")
        assert msg.context_id == "ctx-1"
        assert msg.task_id == "task-1"

    def test_message_id_is_unique(self):
        m1 = _build_message("a")
        m2 = _build_message("b")
        assert m1.message_id != m2.message_id


class TestSerializePart:
    def test_text_part(self):
        result = _serialize_part(make_text_part("hello"))
        assert result == {"kind": "text", "text": "hello"}

    def test_data_part(self):
        result = _serialize_part(make_data_part({"key": "val"}))
        assert result == {"kind": "data", "data": {"key": "val"}}

    def test_file_part_uri(self):
        result = _serialize_part(make_file_part_uri("https://example.com/f.pdf"))
        assert result["kind"] == "file"
        assert result["uri"] == "https://example.com/f.pdf"
        assert result["name"] == "file.pdf"

    def test_file_part_bytes(self):
        result = _serialize_part(make_file_part_bytes("data=="))
        assert result["kind"] == "file"
        assert result["bytes"] == "data=="


# ============================================================================
# A2AClientWrapper tests
# ============================================================================


class TestClientWrapperInit:
    def test_strips_trailing_slash(self):
        c = A2AClientWrapper("http://example.com/")
        assert c._base_url == "http://example.com"

    def test_defaults(self):
        c = A2AClientWrapper("http://example.com")
        assert c._timeout == 30.0
        assert c._headers == {}
        assert c.agent_card is None

    def test_custom_headers(self):
        c = A2AClientWrapper("http://x.com", headers={"X-Key": "abc"})
        assert c._headers == {"X-Key": "abc"}


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_text_response_as_message(self, client_wrapper, mock_a2a_client):
        """Agent returns a direct Message (simple request/response)."""
        response_msg = make_message(
            [make_text_part("I summarized it")],
            context_id="ctx-1",
        )
        mock_a2a_client.send_message.return_value = make_send_success(response_msg)

        result = await client_wrapper.send_message("summarize this")
        assert result.status == "completed"
        assert result.text == "I summarized it"
        assert result.context_id == "ctx-1"

    @pytest.mark.asyncio
    async def test_structured_response_as_message(
        self, client_wrapper, mock_a2a_client
    ):
        """Agent returns a Message with DataPart."""
        response_msg = make_message(
            [make_data_part({"score": 0.95, "label": "positive"})]
        )
        mock_a2a_client.send_message.return_value = make_send_success(response_msg)

        result = await client_wrapper.send_message({"text": "analyze sentiment"})
        assert result.status == "completed"
        assert result.data == [{"score": 0.95, "label": "positive"}]
        assert result.text is None

    @pytest.mark.asyncio
    async def test_task_response_completed(self, client_wrapper, mock_a2a_client):
        """Agent returns a completed Task with text artifact."""
        task = make_task(
            state=TaskState.completed,
            task_id="task-1",
            context_id="ctx-1",
            artifacts=[make_artifact([make_text_part("Summary: ...")])],
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("summarize")
        assert result.status == "completed"
        assert result.task_id == "task-1"
        assert result.context_id == "ctx-1"
        assert result.text == "Summary: ..."
        assert result.requires_input is False

    @pytest.mark.asyncio
    async def test_task_response_with_mixed_artifacts(
        self, client_wrapper, mock_a2a_client
    ):
        """Task with text + data + file artifacts."""
        task = make_task(
            state=TaskState.completed,
            artifacts=[
                make_artifact(
                    [
                        make_text_part("Analysis complete"),
                        make_data_part({"confidence": 0.9}),
                        make_file_part_uri("https://results.com/report.pdf"),
                    ]
                ),
            ],
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("analyze")
        assert result.text == "Analysis complete"
        assert result.data == [{"confidence": 0.9}]
        assert result.files[0]["uri"] == "https://results.com/report.pdf"
        assert len(result.artifacts) == 1
        assert len(result.artifacts[0]["parts"]) == 3

    @pytest.mark.asyncio
    async def test_task_input_required(self, client_wrapper, mock_a2a_client):
        """Task enters input-required state."""
        task = make_task(
            state=TaskState.input_required,
            status_message=make_message([make_text_part("Which language?")]),
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("translate this")
        assert result.status == "input-required"
        assert result.requires_input is True
        assert result.text == "Which language?"

    @pytest.mark.asyncio
    async def test_task_failed(self, client_wrapper, mock_a2a_client):
        """Task enters failed state with error message."""
        task = make_task(
            state=TaskState.failed,
            status_message=make_message([make_text_part("Rate limit exceeded")]),
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("do something")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_task_failed_without_message(self, client_wrapper, mock_a2a_client):
        """Task enters failed state without a status message."""
        task = make_task(state=TaskState.failed)
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("do something")
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_task_rejected(self, client_wrapper, mock_a2a_client):
        """Task enters rejected state."""
        task = make_task(
            state=TaskState.rejected,
            status_message=make_message([make_text_part("Not allowed")]),
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("bad request")
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_jsonrpc_error(self, client_wrapper, mock_a2a_client):
        """JSON-RPC error response raises A2ATaskNotFoundError."""
        from langchain_a2a_adapters.exceptions import A2ATaskNotFoundError

        mock_a2a_client.send_message.return_value = make_send_error(
            -32001, "Task not found"
        )

        with pytest.raises(A2ATaskNotFoundError) as exc_info:
            await client_wrapper.send_message("hello")
        assert exc_info.value.code == -32001
        assert "Task not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_id_forwarded(self, client_wrapper, mock_a2a_client):
        """Context ID is passed through to the message."""
        response_msg = make_message([make_text_part("ok")], context_id="ctx-42")
        mock_a2a_client.send_message.return_value = make_send_success(response_msg)

        result = await client_wrapper.send_message("hi", context_id="ctx-42")
        assert result.context_id == "ctx-42"

    @pytest.mark.asyncio
    async def test_task_with_status_text_fallback(
        self, client_wrapper, mock_a2a_client
    ):
        """Task with no artifacts uses status message as text."""
        task = make_task(
            state=TaskState.completed,
            status_message=make_message([make_text_part("Done processing")]),
        )
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("process")
        assert result.text == "Done processing"
        assert result.artifacts == []

    @pytest.mark.asyncio
    async def test_task_working_state(self, client_wrapper, mock_a2a_client):
        """Task returned in working state."""
        task = make_task(state=TaskState.working)
        mock_a2a_client.send_message.return_value = make_send_success(task)

        result = await client_wrapper.send_message("start job")
        assert result.status == "working"
        assert result.requires_input is False


class TestGetTask:
    @pytest.mark.asyncio
    async def test_success(self, client_wrapper, mock_a2a_client):
        task = make_task(
            state=TaskState.completed,
            task_id="t-1",
            artifacts=[make_artifact([make_text_part("result")])],
        )
        mock_a2a_client.get_task.return_value = make_get_task_success(task)

        result = await client_wrapper.get_task("t-1")
        assert result.status == "completed"
        assert result.task_id == "t-1"
        assert result.text == "result"

    @pytest.mark.asyncio
    async def test_error(self, client_wrapper, mock_a2a_client):
        from langchain_a2a_adapters.exceptions import A2ATaskNotFoundError

        mock_a2a_client.get_task.return_value = make_get_task_error(-32001, "Not found")

        with pytest.raises(A2ATaskNotFoundError) as exc_info:
            await client_wrapper.get_task("t-missing")
        assert exc_info.value.code == -32001
        assert "Not found" in str(exc_info.value)


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_success(self, client_wrapper, mock_a2a_client):
        task = make_task(state=TaskState.canceled, task_id="t-1")
        mock_a2a_client.cancel_task.return_value = make_cancel_task_success(task)

        result = await client_wrapper.cancel_task("t-1")
        assert result.status == "canceled"

    @pytest.mark.asyncio
    async def test_error(self, client_wrapper, mock_a2a_client):
        from langchain_a2a_adapters.exceptions import A2ATaskNotCancelableError

        mock_a2a_client.cancel_task.return_value = make_cancel_task_error(
            -32002, "Not cancelable"
        )

        with pytest.raises(A2ATaskNotCancelableError) as exc_info:
            await client_wrapper.cancel_task("t-1")
        assert exc_info.value.code == -32002
        assert "Not cancelable" in str(exc_info.value)


class TestStreamMessage:
    @pytest.mark.asyncio
    async def test_status_and_artifact_events(self, client_wrapper, mock_a2a_client):
        """Stream yields status and artifact events."""

        async def mock_stream(*args, **kwargs):
            yield make_streaming_status_event(state=TaskState.working)
            yield make_streaming_artifact_event([make_text_part("chunk1")])
            yield make_streaming_artifact_event([make_text_part("chunk2")])
            yield make_streaming_status_event(state=TaskState.completed, final=True)

        mock_a2a_client.send_message_streaming = mock_stream

        events = []
        async for event in client_wrapper.stream_message("hello"):
            events.append(event)

        assert len(events) == 4
        assert events[0].kind == "status-update"
        assert events[1].kind == "artifact-update"
        assert events[2].kind == "artifact-update"
        assert events[3].kind == "status-update"
        assert events[3].final is True

    @pytest.mark.asyncio
    async def test_stream_with_data_artifact(self, client_wrapper, mock_a2a_client):
        """Stream yields artifact with DataPart."""

        async def mock_stream(*args, **kwargs):
            yield make_streaming_artifact_event([make_data_part({"result": 42})])
            yield make_streaming_status_event(state=TaskState.completed, final=True)

        mock_a2a_client.send_message_streaming = mock_stream

        events = []
        async for event in client_wrapper.stream_message("compute"):
            events.append(event)

        # The raw SDK event is yielded; data extraction happens in A2ARunnable
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_stream_error_stops(self, client_wrapper, mock_a2a_client):
        """Streaming raises exception on JSON-RPC error."""
        from langchain_a2a_adapters.exceptions import A2ATaskNotFoundError

        async def mock_stream(*args, **kwargs):
            yield make_streaming_status_event(state=TaskState.working)
            yield make_streaming_error(-32001, "Stream broke")

        mock_a2a_client.send_message_streaming = mock_stream

        events = []
        with pytest.raises(A2ATaskNotFoundError):
            async for event in client_wrapper.stream_message("hello"):
                events.append(event)

        # The first event is yielded; error raised on second
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_stream_dict_input(self, client_wrapper, mock_a2a_client):
        """Stream accepts dict input."""

        async def mock_stream(*args, **kwargs):
            yield make_streaming_status_event(state=TaskState.completed, final=True)

        mock_a2a_client.send_message_streaming = mock_stream

        events = []
        async for event in client_wrapper.stream_message({"action": "run"}):
            events.append(event)

        assert len(events) == 1


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self, client_wrapper):
        """Health check returns True when agent card is available."""
        assert await client_wrapper.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        """Health check returns False when connection fails."""
        wrapper = A2AClientWrapper("http://nonexistent:9999", timeout=0.1)
        assert await wrapper.health_check() is False


class TestTaskLifecycleIntegration:
    """Test full task lifecycle including resubscribe."""

    @pytest.mark.asyncio
    async def test_resubscribe_converts_sdk_events(self, client_wrapper):
        """resubscribe_task() converts SDK events to A2AStreamEvent."""
        from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

        async def mock_resubscribe(request):
            task_id = request.params.id
            yield TaskStatusUpdateEvent(
                task_id=task_id,
                context_id="c1",
                status=TaskStatus(state=TaskState.completed),
                final=True,
            )

        client_wrapper._a2a_client.resubscribe = mock_resubscribe

        events = []
        async for event in client_wrapper.resubscribe_task("task-123"):
            events.append(event)

        assert len(events) == 1
        assert events[0].kind == "status-update"
        assert events[0].task_id == "task-123"
        assert events[0].final is True


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, client_wrapper):
        """Close clears the client references."""
        assert client_wrapper._http_client is not None
        await client_wrapper.close()
        assert client_wrapper._http_client is None
        assert client_wrapper._a2a_client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client_wrapper):
        """Closing twice doesn't raise."""
        await client_wrapper.close()
        await client_wrapper.close()
