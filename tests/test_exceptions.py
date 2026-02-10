"""Tests for exception hierarchy."""

import pytest

from a2a_langchain_adapters.exceptions import (
    A2AAdapterError,
    A2AAuthRequiredError,
    A2ACapabilityError,
    A2AConnectionError,
    A2AContentTypeError,
    A2AProtocolError,
    A2ATaskNotCancelableError,
    A2ATaskNotFoundError,
    A2ATimeoutError,
    A2AUnsupportedOperationError,
    _raise_for_rpc_error,
)


class TestExceptionHierarchy:
    """Test exception inheritance and hierarchy."""

    def test_all_inherit_from_base(self) -> None:
        """All exceptions should inherit from A2AAdapterError."""
        assert issubclass(A2AConnectionError, A2AAdapterError)
        assert issubclass(A2ATimeoutError, A2AAdapterError)
        assert issubclass(A2AProtocolError, A2AAdapterError)
        assert issubclass(A2ATaskNotFoundError, A2AProtocolError)
        assert issubclass(A2ATaskNotCancelableError, A2AProtocolError)
        assert issubclass(A2AUnsupportedOperationError, A2AProtocolError)
        assert issubclass(A2AContentTypeError, A2AProtocolError)
        assert issubclass(A2AAuthRequiredError, A2AAdapterError)
        assert issubclass(A2ACapabilityError, A2AAdapterError)

    def test_exception_instantiation(self) -> None:
        """Test exception creation with and without cause."""
        # Base exception
        exc = A2AAdapterError("Test message")
        assert str(exc) == "Test message"
        assert exc.__cause__ is None

        # With cause
        cause = ValueError("Original error")
        exc = A2AAdapterError("Test message", cause=cause)
        assert str(exc) == "Test message"
        assert exc.__cause__ is cause

    def test_protocol_error_attributes(self) -> None:
        """Test protocol error stores code and data."""
        exc = A2AProtocolError(
            "Test error",
            code=-32001,
            data={"details": "Not found"},
        )
        assert exc.code == -32001
        assert exc.data == {"details": "Not found"}

    def test_protocol_error_with_none_data(self) -> None:
        """Test protocol error with None data."""
        exc = A2AProtocolError("Test error", code=-32001, data=None)
        assert exc.code == -32001
        assert exc.data is None

    def test_specific_protocol_error_codes(self) -> None:
        """Test specific protocol error codes."""
        exc1 = A2ATaskNotFoundError("Not found", code=-32001)
        assert exc1.code == -32001
        assert isinstance(exc1, A2AProtocolError)

        exc2 = A2ATaskNotCancelableError("Not cancelable", code=-32002)
        assert exc2.code == -32002

        exc3 = A2AUnsupportedOperationError("Unsupported", code=-32004)
        assert exc3.code == -32004

        exc4 = A2AContentTypeError("Bad content", code=-32005)
        assert exc4.code == -32005


class TestErrorCodeMapping:
    """Test error code to exception mapping."""

    def test_raise_for_rpc_error_task_not_found(self) -> None:
        """Test raising A2ATaskNotFoundError for code -32001."""

        class MockError:
            code = -32001
            message = "Task not found"
            data = None

        with pytest.raises(A2ATaskNotFoundError) as exc_info:
            _raise_for_rpc_error(MockError())

        assert exc_info.value.code == -32001
        assert "Task not found" in str(exc_info.value)

    def test_raise_for_rpc_error_task_not_cancelable(self) -> None:
        """Test raising A2ATaskNotCancelableError for code -32002."""

        class MockError:
            code = -32002
            message = "Task not cancelable"
            data = None

        with pytest.raises(A2ATaskNotCancelableError) as exc_info:
            _raise_for_rpc_error(MockError())

        assert exc_info.value.code == -32002

    def test_raise_for_rpc_error_unsupported_operation(self) -> None:
        """Test raising A2AUnsupportedOperationError for code -32004."""

        class MockError:
            code = -32004
            message = "Operation not supported"
            data = None

        with pytest.raises(A2AUnsupportedOperationError) as exc_info:
            _raise_for_rpc_error(MockError())

        assert exc_info.value.code == -32004

    def test_raise_for_rpc_error_content_type_error(self) -> None:
        """Test raising A2AContentTypeError for code -32005."""

        class MockError:
            code = -32005
            message = "Content type not supported"
            data = None

        with pytest.raises(A2AContentTypeError) as exc_info:
            _raise_for_rpc_error(MockError())

        assert exc_info.value.code == -32005

    def test_raise_for_rpc_error_unknown_code(self) -> None:
        """Test raising generic A2AProtocolError for unknown code."""

        class MockError:
            code = -32099
            message = "Unknown error"
            data = {"custom": "data"}

        with pytest.raises(A2AProtocolError) as exc_info:
            _raise_for_rpc_error(MockError())

        # Should be base class, not a specific subclass
        assert type(exc_info.value) is A2AProtocolError
        assert exc_info.value.code == -32099
        assert exc_info.value.data == {"custom": "data"}

    def test_raise_for_rpc_error_includes_data(self) -> None:
        """Test that error data is preserved."""

        class MockError:
            code = -32001
            message = "Task not found"
            data = {"task_id": "123", "reason": "expired"}

        with pytest.raises(A2ATaskNotFoundError) as exc_info:
            _raise_for_rpc_error(MockError())

        assert exc_info.value.data == {"task_id": "123", "reason": "expired"}


class TestRetryability:
    """Test error retryability characteristics."""

    def test_transient_errors(self) -> None:
        """Transient errors should be caught for retry."""
        transient_errors = [
            A2AConnectionError("Connection failed"),
            A2ATimeoutError("Request timed out"),
        ]

        for exc in transient_errors:
            assert isinstance(exc, A2AAdapterError)

    def test_non_transient_errors(self) -> None:
        """Non-transient errors should not trigger retry."""
        non_transient = [
            A2ATaskNotFoundError("Not found", code=-32001),
            A2ACapabilityError("Capability not supported"),
            A2AAuthRequiredError("Authentication required"),
        ]

        for exc in non_transient:
            assert isinstance(exc, A2AAdapterError)

    def test_protocol_errors_characteristics(self) -> None:
        """Protocol errors have specific error codes."""
        exc = A2AProtocolError("Error", code=-32000)
        assert hasattr(exc, "code")
        assert hasattr(exc, "data")
        assert exc.code == -32000
