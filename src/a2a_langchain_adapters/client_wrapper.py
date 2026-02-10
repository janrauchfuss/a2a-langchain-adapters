"""Wrapper around a2a-sdk client for LangChain integration."""

from __future__ import annotations

import base64
import logging
import ssl
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    DataPart,
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
    TaskResubscriptionRequest,
    TaskStatusUpdateEvent,
    TextPart,
)

from .auth import A2AAuthConfig
from .exceptions import (
    A2ACapabilityError,
    A2AConnectionError,
    A2ATaskNotFoundError,
    A2ATimeoutError,
    A2AUnsupportedOperationError,
    _raise_for_rpc_error,
)
from .types import A2AResult, A2AStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

logger = logging.getLogger(__name__)

# Public input type for send_message / stream_message
type A2AInput = str | dict[str, Any]


def _extract_parts(
    parts: list[Part],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract typed content from a list of A2A Parts.

    Uses name-based type checking (``type().__name__``) instead of
    ``isinstance()`` to handle version skew between different ``a2a-sdk``
    installations.  When the adapter and a downstream agent pin different
    SDK versions, ``isinstance()`` fails even though the objects are
    structurally identical.

    Returns:
        (texts, data_items, file_items) — one list per Part kind.
    """
    texts: list[str] = []
    data_items: list[dict[str, Any]] = []
    file_items: list[dict[str, Any]] = []

    for part in parts:
        inner = part.root
        if inner is None:
            continue

        type_name = type(inner).__name__

        if type_name == "TextPart" and hasattr(inner, "text"):
            texts.append(inner.text)
        elif type_name == "DataPart" and hasattr(inner, "data"):
            data = inner.data
            if isinstance(data, dict):
                data_items.append(data)
        elif type_name == "FilePart" and hasattr(inner, "file"):
            file_obj = inner.file
            file_info: dict[str, Any] = {}
            # Name-based check for FileWithBytes vs FileWithUri
            if hasattr(file_obj, "bytes") and file_obj.bytes is not None:
                file_info = {
                    "name": file_obj.name,
                    "mime_type": file_obj.mime_type,
                    "bytes": file_obj.bytes,
                }
            else:
                file_info = {
                    "name": file_obj.name,
                    "mime_type": file_obj.mime_type,
                    "uri": file_obj.uri,
                }
            file_items.append(file_info)

    return texts, data_items, file_items


def _extract_text_from_parts(parts: list[Part]) -> str:
    """Extract concatenated text content from a list of A2A Parts."""
    texts, _, _ = _extract_parts(parts)
    return "\n".join(texts)


def _build_message(
    input: A2AInput,
    *,
    files: list[tuple[str, bytes, str]] | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Build a proper A2A Message from string or structured input.

    Supports multiple input formats:
    - ``str``: Creates a single ``TextPart``
    - ``{"text": "...", "data": {...}}``: Mixed text + data parts
    - ``{"data": {...}}``: DataPart only
    - ``{"key": "value", ...}``: Plain dict → single DataPart

    Args:
        input: Text string or dict payload(s).
        files: Optional list of (filename, file_bytes, mime_type) tuples.
        context_id: Conversation context for multi-turn.
        task_id: Existing task to continue.
        metadata: Optional message-level metadata
            (e.g., {"skill": "skill-id"} for routing).

    Returns:
        A2A Message with one or more Parts.

    Raises:
        ValueError: If dict input has no 'text' or 'data' keys and is empty.
    """
    parts: list[Part] = []

    if isinstance(input, str):
        parts.append(Part(root=TextPart(text=input)))
    elif isinstance(input, dict):
        # Check for explicit text/data keys
        has_text_key = "text" in input
        has_data_key = "data" in input

        if has_text_key or has_data_key:
            # Explicit mixed-content format
            if has_text_key:
                parts.append(Part(root=TextPart(text=input["text"])))
            if has_data_key:
                data_val = input["data"]
                # Handle list of dicts or single dict
                items = data_val if isinstance(data_val, list) else [data_val]
                for item in items:
                    parts.append(Part(root=DataPart(data=item)))
        else:
            # Plain dict without text/data keys → single DataPart
            parts.append(Part(root=DataPart(data=input)))

    # Add file parts
    if files:
        for name, content, mime_type in files:
            file_bytes_b64 = base64.b64encode(content).decode("utf-8")
            file_part = Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=file_bytes_b64,
                        mime_type=mime_type,
                        name=name,
                    )
                )
            )
            parts.append(file_part)

    if not parts:
        raise ValueError(
            "Message must contain at least one part (text, data, or files)"
        )

    return Message(
        message_id=str(uuid4()),
        role="user",  # type: ignore[arg-type]
        parts=parts,
        context_id=context_id,
        task_id=task_id,
        metadata=metadata,
    )


def _serialize_part(part: Part) -> dict[str, Any]:
    """Serialize a Part to a plain dict for artifact storage.

    Uses name-based type checking for version-skew resilience.
    """
    inner = part.root
    if inner is None:
        return {"kind": "unknown", "value": "None"}

    type_name = type(inner).__name__

    if type_name == "TextPart" and hasattr(inner, "text"):
        return {"kind": "text", "text": inner.text}
    if type_name == "DataPart" and hasattr(inner, "data"):
        return {"kind": "data", "data": inner.data}
    if type_name == "FilePart" and hasattr(inner, "file"):
        file_obj = inner.file
        if hasattr(file_obj, "bytes") and file_obj.bytes is not None:
            return {
                "kind": "file",
                "name": file_obj.name,
                "mime_type": file_obj.mime_type,
                "bytes": file_obj.bytes,
            }
        return {
            "kind": "file",
            "name": file_obj.name,
            "mime_type": file_obj.mime_type,
            "uri": file_obj.uri,
        }
    return {"kind": "unknown", "value": str(inner)}


class A2AClientWrapper:
    """Wrapper providing LangChain-friendly interface to A2A clients."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
        auth: A2AAuthConfig | None = None,
        transport: str | None = None,
    ) -> None:
        """Initialize A2A client wrapper.

        Args:
            base_url: Agent endpoint URL.
            timeout: Request timeout in seconds (default 30.0).
            headers: Additional HTTP headers.
            auth: Authentication configuration.
            transport: Transport type - None for auto-detect, 'http' for HTTP,
                       'grpc' for gRPC. Auto-detect reads agent card's
                       preferred_transport field.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._auth = auth
        self._transport = transport

        # Merge auth headers with provided headers
        self._headers = dict(headers or {})
        if auth:
            self._headers.update(auth.build_headers())

        self._http_client: httpx.AsyncClient | None = None
        self._a2a_client: A2AClient | None = None
        self._agent_card: AgentCard | None = None

    def _build_transport(self) -> Any:
        """Build transport instance based on configuration.

        Returns:
            Transport instance or None for auto-detect.

        Raises:
            ValueError: If transport is invalid.
            ImportError: If gRPC transport requested but grpcio not installed.
        """
        if self._transport is None or self._transport == "http":
            # HTTP transport or auto-detect (defaults to HTTP for now)
            # For now, keep using httpx.AsyncClient
            return None

        if self._transport == "grpc":
            try:
                # Import here to avoid hard dependency
                import grpc

                # Build gRPC channel with auth
                channel_credentials = None
                if self._auth and self._auth.tls_certificates:
                    # Load mTLS credentials for gRPC
                    with open(self._auth.tls_certificates.client_cert_path, "rb") as f:
                        cert = f.read()
                    with open(self._auth.tls_certificates.client_key_path, "rb") as f:
                        key = f.read()

                    ca_cert = None
                    if self._auth.tls_certificates.ca_cert_path:
                        with open(self._auth.tls_certificates.ca_cert_path, "rb") as f:
                            ca_cert = f.read()

                    channel_credentials = grpc.ssl_channel_credentials(
                        root_certificates=ca_cert,
                        private_key=key,
                        certificate_chain=cert,
                    )

                return channel_credentials

            except ImportError as e:
                raise ImportError(
                    "gRPC transport requires the 'grpc' extra. "
                    "Install with: pip install a2a-langchain-adapters[grpc]"
                ) from e

        raise ValueError(
            f"Unsupported transport: {self._transport}. "
            "Supported: None (auto), 'http', 'grpc'"
        )

    async def _ensure_client(self) -> A2AClient:
        """Lazily initialize the A2A client via agent card discovery.

        Currently supports HTTP transport. gRPC transport support is
        available via the [grpc] extra and transport parameter.
        """
        if self._a2a_client is None:
            # Log transport choice
            actual_transport = self._transport or "auto-detect"
            logger.debug(
                "Initializing A2A client with %s transport at %s",
                actual_transport,
                self._base_url,
            )

            # Validate transport parameter if needed
            if self._transport and self._transport not in ("http", "grpc"):
                raise ValueError(
                    f"Unsupported transport: {self._transport}. "
                    "Supported: None (auto), 'http', 'grpc'"
                )

            # Build SSL context for mTLS if configured
            verify: bool | ssl.SSLContext = True
            if self._auth:
                ssl_context = self._auth.build_ssl_context()
                if ssl_context:
                    verify = ssl_context

            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers,
                verify=verify,
            )
            resolver = A2ACardResolver(self._http_client, self._base_url)
            self._agent_card = await resolver.get_agent_card()
            self._a2a_client = A2AClient(self._http_client, self._agent_card)
        return self._a2a_client

    @property
    def agent_card(self) -> AgentCard | None:
        return self._agent_card

    async def get_agent_card(self) -> AgentCard:
        await self._ensure_client()
        assert self._agent_card is not None
        return self._agent_card

    def get_security_schemes(self) -> dict[str, Any] | None:
        """Get security schemes from agent card.

        Returns:
            Dictionary of security schemes, or None if not advertised.
        """
        if not self._agent_card:
            return None

        schemes = {}
        if hasattr(self._agent_card, "security_schemes"):
            security_schemes = getattr(self._agent_card, "security_schemes", None)
            if security_schemes:
                for name, scheme in security_schemes.items():
                    scheme_dict = (
                        scheme
                        if isinstance(scheme, dict)
                        else getattr(scheme, "__dict__", {})
                    )
                    schemes[name] = scheme_dict

        return schemes if schemes else None

    def requires_mTLS(self) -> bool:
        """Check if agent requires mTLS.

        Returns:
            True if agent advertises mutualTLS security scheme.
        """
        schemes = self.get_security_schemes()
        if not schemes:
            return False

        for scheme in schemes.values():
            scheme_type = (
                scheme.get("type")
                if isinstance(scheme, dict)
                else getattr(scheme, "type", None)
            )
            if scheme_type == "mutualTLS":
                return True

        return False

    async def download_file(
        self,
        file_uri: str,
        save_path: str | None = None,
    ) -> bytes:
        """Download file from URI in A2AResult.

        Args:
            file_uri: URI from A2AResult.files[*]["uri"].
            save_path: Optional path to save file locally.

        Returns:
            Downloaded file bytes.

        Raises:
            httpx.HTTPError: If download fails.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(file_uri)
            response.raise_for_status()
            content = response.content

        if save_path:
            from pathlib import Path

            Path(save_path).write_bytes(content)

        return content

    def decode_file_bytes(self, file_data: dict[str, Any]) -> bytes:
        """Decode base64 file bytes from A2AResult.

        Args:
            file_data: File entry from A2AResult.files[*].

        Returns:
            Decoded file bytes.

        Raises:
            ValueError: If file_data has no 'bytes' field.
        """
        if "bytes" not in file_data:
            raise ValueError(
                "File data has no 'bytes' field. Use download_file() "
                "for files with 'uri' instead."
            )
        return base64.b64decode(file_data["bytes"])

    def _check_capability(self, capability: str) -> None:
        """Raise A2ACapabilityError if agent doesn't support capability.

        Only raises if the agent explicitly marks the capability as False.
        Permits None (not declared) and True (supported).

        Args:
            capability: One of 'streaming', 'push_notifications',
                'state_transition_history'.

        Raises:
            A2ACapabilityError: If agent does not support the capability.
        """
        if self._agent_card is None:
            return  # Card not yet loaded
        caps = self._agent_card.capabilities
        if caps is None:
            return  # No capabilities declared; assume supported

        capability_map = {
            "streaming": caps.streaming,
            "push_notifications": caps.push_notifications,
            "state_transition_history": caps.state_transition_history,
        }
        supported = capability_map.get(capability)
        if supported is False:
            raise A2ACapabilityError(
                f"Agent '{self._agent_card.name}' does not support "
                f"'{capability}'. Advertised capabilities: {caps}"
            )

    def _check_input_modes(self, content_types: list[str]) -> None:
        """Warn if content types not in default_input_modes.

        Logs a warning if any content type is not listed in the agent's
        declared input modes. Does not raise (soft check).

        Args:
            content_types: List of content type strings (e.g.,
                ['application/json']).
        """
        if self._agent_card is None:
            return
        accepted = self._agent_card.default_input_modes or ["text/plain"]
        for ct in content_types:
            if ct not in accepted:
                logger.warning(
                    "Agent '%s' does not list '%s' in default_input_modes %s. "
                    "The request may be rejected.",
                    self._agent_card.name,
                    ct,
                    accepted,
                )

    async def send_message(
        self,
        input: A2AInput,
        *,
        files: list[tuple[str, bytes, str]] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> A2AResult:
        """Send a message to the A2A agent and return a structured result.

        Args:
            input: Text string or structured dict payload.
            files: Optional list of (filename, file_bytes, mime_type) tuples.
            context_id: Conversation context for multi-turn.
            task_id: Existing task to continue.
            metadata: Optional message-level metadata
                (e.g., {"skill": "skill-id"} for routing).

        Returns:
            A2AResult with all Part types preserved.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2ATimeoutError: If request times out.
            A2AProtocolError: If agent returns JSON-RPC error.
        """
        # Auto-wrap string input for DataPart-only agents
        if isinstance(input, str):
            modes = self._agent_card.default_input_modes if self._agent_card else None
            if modes and "text/plain" not in modes:
                # Agent only accepts structured input, wrap as query
                input = {"query": input}

        # Check input modes if sending data
        if isinstance(input, dict) and "data" in input:
            self._check_input_modes(["application/json"])

        client = await self._ensure_client()

        message = _build_message(
            input,
            files=files,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message=message,
                configuration=MessageSendConfiguration(blocking=True),
            ),
        )

        logger.debug("Sending A2A message to %s", self._base_url)
        try:
            response = await client.send_message(request)
        except httpx.ConnectError as e:
            raise A2AConnectionError(
                f"Failed to connect to A2A agent at {self._base_url}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise A2ATimeoutError(
                f"Request to A2A agent {self._base_url} timed out after "
                f"{self._timeout}s",
                cause=e,
            ) from e

        root = response.root

        if isinstance(root, JSONRPCErrorResponse):
            _raise_for_rpc_error(root.error)

        result = root.result

        # Response is a direct Message (simple request/response)
        if isinstance(result, Message):
            texts, data_items, file_items = _extract_parts(result.parts)
            return A2AResult(
                task_id=task_id or "",
                context_id=result.context_id or context_id or "",
                status="completed",
                text="\n".join(texts) if texts else None,
                data=data_items,
                files=file_items,
            )

        # Response is a Task (full lifecycle)
        if isinstance(result, Task):
            return self._task_to_result(result)

        return A2AResult(
            task_id="",
            context_id="",
            status="unknown",
        )

    def _task_to_result(self, task: Task) -> A2AResult:
        """Convert an A2A Task to an A2AResult."""
        state = task.status.state.value

        # Extract content from artifacts
        all_texts: list[str] = []
        all_data: list[dict[str, Any]] = []
        all_files: list[dict[str, Any]] = []
        artifacts_raw: list[dict[str, Any]] = []

        for artifact in task.artifacts or []:
            texts, data_items, file_items = _extract_parts(artifact.parts)
            all_texts.extend(texts)
            all_data.extend(data_items)
            all_files.extend(file_items)
            artifacts_raw.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "parts": [_serialize_part(p) for p in artifact.parts],
                }
            )

        # Extract ALL part types from status message (not just text)
        status_text = None
        if task.status.message:
            texts, data_items, file_items = _extract_parts(task.status.message.parts)
            status_text = "\n".join(texts) if texts else None
            all_data.extend(data_items)
            all_files.extend(file_items)

        # Fallback: if no data in artifacts or status, check last agent
        # message in history
        if not all_data and task.history:
            for msg in reversed(task.history):
                if getattr(msg, "role", None) == "agent":
                    _, data_items, file_items = _extract_parts(msg.parts)
                    if data_items:
                        all_data.extend(data_items)
                        if not all_files:
                            all_files.extend(file_items)
                        break

        text = "\n".join(all_texts) if all_texts else status_text

        return A2AResult(
            task_id=task.id,
            context_id=task.context_id,
            status=state,
            text=text,
            data=all_data,
            files=all_files,
            artifacts=artifacts_raw,
            requires_input=(state == "input-required"),
        )

    async def get_task(
        self, task_id: str, *, history_length: int | None = None
    ) -> A2AResult:
        """Retrieve the current state of a task via tasks/get.

        Args:
            task_id: Task ID to retrieve.
            history_length: Number of messages to include in history.

        Returns:
            A2AResult with current task state.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2ATimeoutError: If request times out.
            A2AProtocolError: If agent returns JSON-RPC error.
        """
        client = await self._ensure_client()
        try:
            response = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=task_id, history_length=history_length),
                )
            )
        except httpx.ConnectError as e:
            raise A2AConnectionError(
                f"Failed to connect to A2A agent at {self._base_url}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise A2ATimeoutError(
                f"Request to A2A agent {self._base_url} timed out after "
                f"{self._timeout}s",
                cause=e,
            ) from e

        root = response.root

        if isinstance(root, JSONRPCErrorResponse):
            _raise_for_rpc_error(root.error)

        return self._task_to_result(root.result)

    async def cancel_task(self, task_id: str) -> A2AResult:
        """Cancel a running task via tasks/cancel.

        Args:
            task_id: Task ID to cancel.

        Returns:
            A2AResult with updated task state (should be 'canceled').

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2ATimeoutError: If request times out.
            A2AProtocolError: If agent returns JSON-RPC error.
        """
        client = await self._ensure_client()
        try:
            response = await client.cancel_task(
                CancelTaskRequest(
                    id=str(uuid4()),
                    params=TaskIdParams(id=task_id),
                )
            )
        except httpx.ConnectError as e:
            raise A2AConnectionError(
                f"Failed to connect to A2A agent at {self._base_url}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise A2ATimeoutError(
                f"Request to A2A agent {self._base_url} timed out after "
                f"{self._timeout}s",
                cause=e,
            ) from e

        root = response.root

        if isinstance(root, JSONRPCErrorResponse):
            _raise_for_rpc_error(root.error)

        return self._task_to_result(root.result)

    async def resubscribe_task(
        self,
        task_id: str,
        *,
        context_id: str | None = None,
    ) -> AsyncIterator[A2AStreamEvent]:
        """Reconnect to an interrupted streaming task.

        Use this to resume streaming for long-running tasks after network
        interruption or client restart.

        Args:
            task_id: ID of task to resubscribe to.
            context_id: Reserved for future use (not currently used by SDK).

        Yields:
            A2AStreamEvent instances as task progresses.

        Raises:
            A2ATaskNotFoundError: If task_id not found or expired.
            A2AUnsupportedOperationError: If task is not streamable.
            A2AConnectionError: If network fails.

        Example:
            ```python
            # Initial streaming interrupted
            task_id = "abc-123"

            # Later, reconnect to same task
            async for event in wrapper.resubscribe_task(task_id):
                print(f"Resumed: {event.text}")
            ```
        """
        try:
            client = await self._ensure_client()

            # Build resubscription request with task ID
            request = TaskResubscriptionRequest(
                id=str(uuid4()),
                params=TaskIdParams(id=task_id),
            )

            async for event in client.resubscribe(request):
                # Convert SDK events to A2AStreamEvent
                if isinstance(event, TaskStatusUpdateEvent):
                    # Extract data/text from status message if present
                    status_text = None
                    status_data: list[dict[str, Any]] = []
                    if event.status.message:
                        texts, data_items, _ = _extract_parts(
                            event.status.message.parts
                        )
                        status_text = "\n".join(texts) if texts else None
                        status_data = data_items
                    yield A2AStreamEvent(
                        kind="status-update",
                        task_id=event.task_id,
                        context_id=event.context_id,
                        status=event.status.state.value
                        if hasattr(event.status.state, "value")
                        else str(event.status.state),
                        text=status_text,
                        data=status_data,
                        final=event.final,
                    )

                elif isinstance(event, TaskArtifactUpdateEvent):
                    text, data_items, _ = _extract_parts(event.artifact.parts)

                    yield A2AStreamEvent(
                        kind="artifact-update",
                        task_id=event.task_id,
                        context_id=event.context_id,
                        text="\n".join(text) if text else None,
                        data=data_items,
                    )

        except Exception as e:
            # Wrap SDK exceptions
            if "not found" in str(e).lower():
                raise A2ATaskNotFoundError(
                    f"Task {task_id} not found or expired",
                    code=-32001,
                    cause=e,
                ) from e

            if "not streamable" in str(e).lower():
                raise A2AUnsupportedOperationError(
                    f"Task {task_id} does not support resubscribe",
                    code=-32004,
                    cause=e,
                ) from e

            raise A2AConnectionError(
                f"Failed to resubscribe to task {task_id}",
                cause=e,
            ) from e

    async def stream_message(
        self,
        input: A2AInput,
        *,
        files: list[tuple[str, bytes, str]] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[A2AStreamEvent]:
        """Stream a message response via message/stream (SSE).

        Yields A2AStreamEvent objects with extracted text and data.

        Args:
            input: Text string or structured dict payload.
            files: Optional list of (filename, file_bytes, mime_type) tuples.
            context_id: Conversation context for multi-turn.
            task_id: Existing task to continue.
            metadata: Optional message-level metadata
                (e.g., {"skill": "skill-id"} for routing).

        Yields:
            A2AStreamEvent with status updates and artifact updates.

        Raises:
            A2ACapabilityError: If agent does not support streaming.
        """
        # Check capability before making request
        self._check_capability("streaming")

        # Auto-wrap string input for DataPart-only agents
        if isinstance(input, str):
            modes = self._agent_card.default_input_modes if self._agent_card else None
            if modes and "text/plain" not in modes:
                # Agent only accepts structured input, wrap as query
                input = {"query": input}

        client = await self._ensure_client()

        message = _build_message(
            input,
            files=files,
            context_id=context_id,
            task_id=task_id,
            metadata=metadata,
        )
        request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )

        try:
            async for event in client.send_message_streaming(request):
                root = event.root

                if isinstance(root, JSONRPCErrorResponse):
                    _raise_for_rpc_error(root.error)

                result = root.result
                if isinstance(result, TaskStatusUpdateEvent):
                    # Extract data/text from status message if present
                    status_text = None
                    status_data: list[dict[str, Any]] = []
                    if result.status.message:
                        texts, data_items, _ = _extract_parts(
                            result.status.message.parts
                        )
                        status_text = "\n".join(texts) if texts else None
                        status_data = data_items
                    yield A2AStreamEvent(
                        kind="status-update",
                        task_id=result.task_id,
                        context_id=result.context_id,
                        status=result.status.state.value
                        if hasattr(result.status.state, "value")
                        else str(result.status.state),
                        text=status_text,
                        data=status_data,
                        final=result.final,
                    )
                    if result.final:
                        return
                elif isinstance(result, TaskArtifactUpdateEvent):
                    texts, data_items, _ = _extract_parts(result.artifact.parts)
                    yield A2AStreamEvent(
                        kind="artifact-update",
                        task_id=result.task_id,
                        context_id=result.context_id,
                        text="\n".join(texts) if texts else None,
                        data=data_items,
                    )
        except httpx.ConnectError as e:
            raise A2AConnectionError(
                f"Failed to connect to A2A agent at {self._base_url}",
                cause=e,
            ) from e
        except httpx.TimeoutException as e:
            raise A2ATimeoutError(
                f"Request to A2A agent {self._base_url} timed out after "
                f"{self._timeout}s",
                cause=e,
            ) from e

    async def health_check(self) -> bool:
        """Check if the A2A agent is reachable by fetching its agent card."""
        try:
            await self.get_agent_card()
        except Exception:
            logger.warning(
                "A2A health check failed for %s", self._base_url, exc_info=True
            )
            return False
        else:
            return True

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self._a2a_client = None
