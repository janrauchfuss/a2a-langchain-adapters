"""Exception hierarchy for a2a-langchain-adapters.

Provides typed exceptions for A2A adapter errors, enabling programmatic
error handling and retry strategies based on error type.
"""

from __future__ import annotations

from typing import Any, NoReturn


class A2AAdapterError(Exception):
    """Base exception for all a2a-langchain-adapters errors.

    All A2A adapter exceptions inherit from this base class for easy
    catching and error handling. Supports optional cause chaining.

    Attributes:
        message: Human-readable error message.
        __cause__: Optional chained exception (from another error).
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        """Initialize A2A adapter error.

        Args:
            message: Error description.
            cause: Optional exception that caused this error.
        """
        super().__init__(message)
        self.__cause__ = cause


class A2AConnectionError(A2AAdapterError):
    """Connection to A2A agent failed.

    Raised when the adapter cannot establish a connection to the A2A agent.
    This includes network errors, DNS resolution failures, and TLS/SSL errors.

    **Retryable:** Yes - typically a transient issue.
    """

    pass


class A2ATimeoutError(A2AAdapterError):
    """Request to A2A agent timed out.

    Raised when a request to the A2A agent exceeds the configured timeout.
    The server may be overloaded or the network may be slow.

    **Retryable:** Yes - likely a transient issue.
    """

    pass


class A2AProtocolError(A2AAdapterError):
    """A2A protocol-level error (JSON-RPC error response).

    The A2A agent returned a JSON-RPC error response. This includes both
    standard JSON-RPC errors (-32000 to -32099) and A2A-specific errors
    (-32001 to -32006).

    Attributes:
        code: JSON-RPC error code.
        data: Optional error data from the server (dict).

    **Retryable:** Depends on error code. See specific subclasses.
    """

    def __init__(
        self,
        message: str,
        *,
        code: int,
        data: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize protocol error.

        Args:
            message: Error description.
            code: JSON-RPC error code.
            data: Optional error data from server.
            cause: Optional chained exception.
        """
        super().__init__(message, cause=cause)
        self.code = code
        self.data = data


class A2ATaskNotFoundError(A2AProtocolError):
    """Task ID not found on the A2A server (code -32001).

    The `task_id` provided in a `tasks/get`, `tasks/cancel`, or related
    operation does not exist on the server.

    **Retryable:** No - the task is permanently not found.
    """

    pass


class A2ATaskNotCancelableError(A2AProtocolError):
    """Task cannot be canceled in its current state (code -32002).

    The task exists but is in a terminal state (completed, failed, canceled,
    etc.) and cannot be canceled.

    **Retryable:** No - the task state is final.
    """

    pass


class A2AUnsupportedOperationError(A2AProtocolError):
    """A2A agent does not support the requested operation (code -32004).

    The agent returned an error indicating it does not support the operation
    being requested (e.g., streaming when agent.capabilities.streaming is
    False).

    **Retryable:** No - the agent will never support this operation.
    """

    pass


class A2AContentTypeError(A2AProtocolError):
    """Content type not supported by the A2A agent (code -32005).

    The agent rejected the message content type or format. Common causes:
    - Sending `DataPart` when agent only accepts text
    - Sending `FilePart` when agent doesn't support files

    **Retryable:** No - the agent configuration is fixed.
    """

    pass


class A2AAuthRequiredError(A2AAdapterError):
    """A2A agent requires authentication.

    The agent returned an `auth-required` task state, indicating that
    credentials are missing, invalid, or have expired. Configure
    authentication via `A2AAuthConfig` (v0.2) or pass credentials in
    `headers` (v0.1).

    **Retryable:** No - requires user intervention to provide credentials.
    """

    pass


class A2ACapabilityError(A2AAdapterError):
    """A2A agent does not advertise a required capability.

    Raised when trying to use a capability that the agent has explicitly
    marked as unsupported in its agent card. Examples:
    - Calling `.astream()` on an agent with `capabilities.streaming == False`
    - Sending a `DataPart` to an agent with limited `default_input_modes`

    **Retryable:** No - the agent configuration is fixed.
    """

    pass


# Error code mapping for JSON-RPC error responses
_ERROR_CODE_MAP: dict[int, type[A2AProtocolError]] = {
    -32001: A2ATaskNotFoundError,
    -32002: A2ATaskNotCancelableError,
    -32004: A2AUnsupportedOperationError,
    -32005: A2AContentTypeError,
}


def _raise_for_rpc_error(error: Any) -> NoReturn:
    """Convert a JSON-RPC error response to a typed A2A exception.

    Maps error codes to specific exception types and raises the appropriate
    exception with the error code and data included.

    Args:
        error: A JSONRPCError object from the a2a-sdk.

    Raises:
        A2AProtocolError: Or one of its subclasses based on error code.

    Example:
        ```python
        response = await client.send_message(request)
        if hasattr(response, "error") and response.error:
            _raise_for_rpc_error(response.error)
        ```
    """
    error_code = error.code
    error_message = error.message
    error_data = getattr(error, "data", None)

    # Look up specific exception class for this code
    exc_class = _ERROR_CODE_MAP.get(error_code, A2AProtocolError)

    # Raise with code and data
    raise exc_class(
        f"[{error_code}] {error_message}",
        code=error_code,
        data=error_data,
    )
