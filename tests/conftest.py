"""Shared fixtures for a2a-langchain-adapters tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    CancelTaskSuccessResponse,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    GetTaskSuccessResponse,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    Part,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from a2a_langchain_adapters.client_wrapper import A2AClientWrapper

# ---------------------------------------------------------------------------
# Agent cards
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_card() -> AgentCard:
    """An AgentCard with multiple skills."""
    return AgentCard(
        name="Test Agent",
        description="A test agent for unit tests",
        url="http://test-agent:8080",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="summarize",
                name="Summarize",
                description="Summarize documents",
                tags=["summarize"],
            ),
            AgentSkill(
                id="translate",
                name="Translate",
                description="Translate between languages",
                tags=["translate"],
            ),
        ],
    )


@pytest.fixture
def agent_card_no_skills() -> AgentCard:
    """An AgentCard with no skills."""
    return AgentCard(
        name="Simple Agent",
        description="A simple agent",
        url="http://simple-agent:8080",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )


# ---------------------------------------------------------------------------
# A2A Part helpers
# ---------------------------------------------------------------------------


def make_text_part(text: str) -> Part:
    return Part(root=TextPart(text=text))


def make_data_part(data: dict[str, object]) -> Part:
    return Part(root=DataPart(data=data))


def make_file_part_uri(
    uri: str, *, name: str = "file.pdf", mime_type: str = "application/pdf"
) -> Part:
    return Part(
        root=FilePart(file=FileWithUri(uri=uri, name=name, mime_type=mime_type))
    )


def make_file_part_bytes(
    content: str, *, name: str = "file.txt", mime_type: str = "text/plain"
) -> Part:
    return Part(
        root=FilePart(file=FileWithBytes(bytes=content, name=name, mime_type=mime_type))
    )


# ---------------------------------------------------------------------------
# A2A Message / Task builders
# ---------------------------------------------------------------------------


def make_message(
    parts: list[Part],
    *,
    role: str = "agent",
    context_id: str | None = None,
) -> Message:
    return Message(
        message_id=str(uuid4()),
        role=role,  # type: ignore[arg-type]
        parts=parts,
        context_id=context_id,
    )


def make_task(
    *,
    state: TaskState = TaskState.completed,
    artifacts: list[Artifact] | None = None,
    status_message: Message | None = None,
    history: list[Message] | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> Task:
    return Task(
        id=task_id or str(uuid4()),
        context_id=context_id or str(uuid4()),
        status=TaskStatus(state=state, message=status_message),
        artifacts=artifacts,
        history=history,
    )


def make_artifact(parts: list[Part], *, name: str | None = None) -> Artifact:
    return Artifact(
        artifact_id=str(uuid4()),
        parts=parts,
        name=name,
    )


# ---------------------------------------------------------------------------
# JSON-RPC response builders
# ---------------------------------------------------------------------------


def make_send_success(result: Task | Message) -> SendMessageResponse:
    return SendMessageResponse(root=SendMessageSuccessResponse(id="1", result=result))


def make_send_error(
    code: int = -32001, message: str = "Task not found"
) -> SendMessageResponse:
    return SendMessageResponse(
        root=JSONRPCErrorResponse(
            id="1", error=JSONRPCError(code=code, message=message)
        )
    )


def make_get_task_success(task: Task) -> MagicMock:
    """Build a GetTaskResponse-like object."""
    resp = MagicMock()
    resp.root = GetTaskSuccessResponse(id="1", result=task)
    return resp


def make_get_task_error(
    code: int = -32001, message: str = "Task not found"
) -> MagicMock:
    resp = MagicMock()
    resp.root = JSONRPCErrorResponse(
        id="1", error=JSONRPCError(code=code, message=message)
    )
    return resp


def make_cancel_task_success(task: Task) -> MagicMock:
    resp = MagicMock()
    resp.root = CancelTaskSuccessResponse(id="1", result=task)
    return resp


def make_cancel_task_error(
    code: int = -32002, message: str = "Not cancelable"
) -> MagicMock:
    resp = MagicMock()
    resp.root = JSONRPCErrorResponse(
        id="1", error=JSONRPCError(code=code, message=message)
    )
    return resp


def make_streaming_status_event(
    *,
    task_id: str = "t1",
    context_id: str = "c1",
    state: TaskState = TaskState.working,
    status_message: Message | None = None,
    final: bool = False,
) -> SendStreamingMessageResponse:
    return SendStreamingMessageResponse(
        root=SendStreamingMessageSuccessResponse(
            id="1",
            result=TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=state, message=status_message),
                final=final,
            ),
        )
    )


def make_streaming_artifact_event(
    parts: list[Part],
    *,
    task_id: str = "t1",
    context_id: str = "c1",
    name: str | None = None,
) -> SendStreamingMessageResponse:
    return SendStreamingMessageResponse(
        root=SendStreamingMessageSuccessResponse(
            id="1",
            result=TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=Artifact(
                    artifact_id=str(uuid4()),
                    parts=parts,
                    name=name,
                ),
            ),
        )
    )


def make_streaming_error(
    code: int = -32001, message: str = "Stream error"
) -> SendStreamingMessageResponse:
    return SendStreamingMessageResponse(
        root=JSONRPCErrorResponse(
            id="1", error=JSONRPCError(code=code, message=message)
        )
    )


# ---------------------------------------------------------------------------
# Pre-wired client wrapper fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_a2a_client() -> AsyncMock:
    """A mocked a2a.client.A2AClient."""
    return AsyncMock()


@pytest.fixture
def agent_card_with_examples() -> AgentCard:
    """AgentCard with skills that have examples and tags."""
    return AgentCard(
        name="Enhanced Agent",
        description="Agent with rich metadata",
        url="http://enhanced-agent:8080",
        version="1.0.0",
        protocol_version="1.0.0",
        documentation_url="https://docs.example.com/agent",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="summarize",
                name="Summarize",
                description="Summarize documents",
                tags=["nlp", "summarization"],
                examples=[
                    "Summarize the Q4 earnings report",
                    "Create a brief summary of the meeting notes",
                ],
            ),
        ],
    )


@pytest.fixture
def client_wrapper(
    agent_card: AgentCard, mock_a2a_client: AsyncMock
) -> A2AClientWrapper:
    """An A2AClientWrapper with pre-injected mocks (no HTTP needed)."""
    wrapper = A2AClientWrapper("http://test-agent:8080")
    wrapper._agent_card = agent_card
    wrapper._a2a_client = mock_a2a_client
    wrapper._http_client = AsyncMock()  # prevent real HTTP
    return wrapper
