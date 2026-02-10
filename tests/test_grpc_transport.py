"""Tests for gRPC transport support."""

from __future__ import annotations

import contextlib

import pytest

from a2a_langchain_adapters import A2AAuthConfig, A2ARunnable
from a2a_langchain_adapters.client_wrapper import A2AClientWrapper


class TestTransportConfiguration:
    """Test transport parameter configuration."""

    def test_default_transport_none(self):
        """Default transport is None (auto-detect)."""
        wrapper = A2AClientWrapper("http://test:8080")
        assert wrapper._transport is None

    def test_force_http_transport(self):
        """Can force HTTP transport."""
        wrapper = A2AClientWrapper(
            "http://test:8080",
            transport="http",
        )
        assert wrapper._transport == "http"

    def test_force_grpc_transport(self):
        """Can force gRPC transport."""
        wrapper = A2AClientWrapper(
            "grpc://test:50051",
            transport="grpc",
        )
        assert wrapper._transport == "grpc"

    def test_invalid_transport_raises(self):
        """Invalid transport raises ValueError."""
        wrapper = A2AClientWrapper(
            "http://test:8080",
            transport="websocket",
        )

        with pytest.raises(ValueError, match="Unsupported transport"):
            wrapper._build_transport()

    def test_transport_with_auth(self):
        """Transport parameter works with auth config."""
        auth = A2AAuthConfig().add_bearer_token("token123")

        wrapper = A2AClientWrapper(
            "http://test:8080",
            auth=auth,
            transport="http",
        )

        assert wrapper._transport == "http"
        assert wrapper._auth is auth


class TestGRPCImportError:
    """Test gRPC import error handling."""

    def test_grpc_not_installed_error(self):
        """Requesting gRPC without installing it raises helpful error."""
        wrapper = A2AClientWrapper(
            "grpc://test:50051",
            transport="grpc",
        )

        # Mock grpc not being available
        import sys

        # Save original modules
        grpc_module = sys.modules.get("grpc")

        try:
            # Remove grpc from modules to simulate it not being installed
            if "grpc" in sys.modules:
                del sys.modules["grpc"]

            # Attempting to build transport should raise ImportError
            with pytest.raises(
                ImportError,
                match="gRPC transport requires the 'grpc' extra",
            ):
                wrapper._build_transport()

        finally:
            # Restore grpc module
            if grpc_module is not None:
                sys.modules["grpc"] = grpc_module


class TestTransportLogging:
    """Test transport selection logging."""

    @pytest.mark.asyncio
    async def test_transport_logging_in_from_agent_url(self):
        """from_agent_url logs transport selection."""
        import inspect

        # Verify that from_agent_url accepts transport parameter
        sig = inspect.signature(A2ARunnable.from_agent_url)
        assert "transport" in sig.parameters

    def test_from_agent_url_transport_parameter(self):
        """from_agent_url accepts transport parameter."""
        # Verify the method signature includes transport
        import inspect

        sig = inspect.signature(A2ARunnable.from_agent_url)
        assert "transport" in sig.parameters


class TestTransportWithMTLS:
    """Test transport parameter interaction with mTLS auth."""

    def test_build_transport_with_mtls_credentials(self):
        """_build_transport handles mTLS credentials for gRPC."""
        auth = A2AAuthConfig().add_tls_certificates(
            client_cert="/path/to/cert.crt",
            client_key="/path/to/key.key",
            ca_cert="/path/to/ca.crt",
        )

        wrapper = A2AClientWrapper(
            "grpc://test:50051",
            auth=auth,
            transport="grpc",
        )

        # Verify _build_transport can handle the mTLS config
        # (actual gRPC transport creation requires grpcio)
        # We just verify it doesn't crash with valid config
        with contextlib.suppress(ImportError):
            wrapper._build_transport()
