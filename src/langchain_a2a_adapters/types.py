"""Type definitions for langchain-a2a-adapters."""

from typing import Any

from pydantic import BaseModel, Field


class A2AResult(BaseModel):
    """Structured result from an A2A agent invocation.

    Returned by A2ARunnable.invoke() / .ainvoke(). Preserves full
    protocol semantics including task state and conversation context.

    Content is split by Part type:
    - ``text``: concatenated TextPart content
    - ``data``: list of DataPart payloads (structured JSON)
    - ``files``: list of FilePart references (name, mime_type, uri/bytes)
    - ``artifacts``: raw artifact-level view (all part types per artifact)
    """

    task_id: str
    context_id: str
    status: str  # TaskState value
    text: str | None = None
    data: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    requires_input: bool = False  # True when status is input-required


class A2AStreamEvent(BaseModel):
    """A single streaming event from an A2A agent.

    Yielded by A2ARunnable.astream(). Wraps TaskStatusUpdateEvent
    and TaskArtifactUpdateEvent into a LangChain-friendly format.
    """

    kind: str  # "status-update" or "artifact-update"
    task_id: str
    context_id: str
    text: str | None = None  # extracted text (if artifact with TextPart)
    data: list[dict[str, Any]] = Field(default_factory=list)  # DataPart payloads
    status: str | None = None  # TaskState (if status event)
    final: bool = False  # True = stream terminated
