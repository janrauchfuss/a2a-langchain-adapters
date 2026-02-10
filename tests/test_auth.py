"""Tests for authentication configuration and integration."""

from __future__ import annotations

import base64
import ssl
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_langchain_adapters import (
    A2AAuthConfig,
    A2ARunnable,
    APIKeyCredentials,
    BasicAuthCredentials,
    BearerTokenCredentials,
    TLSCertificates,
)
from a2a_langchain_adapters.client_wrapper import A2AClientWrapper


class TestBearerTokenAuth:
    """Test Bearer token authentication."""

    def test_bearer_token_header(self):
        """Bearer token generates correct Authorization header."""
        config = A2AAuthConfig()
        config.add_bearer_token("eyJhbGc...")

        headers = config.build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer eyJhbGc..."

    def test_bearer_token_custom_type(self):
        """Bearer token with custom type."""
        config = A2AAuthConfig()
        config.add_bearer_token("token123", token_type="DPoP")

        headers = config.build_headers()

        assert headers["Authorization"] == "DPoP token123"

    def test_bearer_token_persistence(self):
        """Bearer token configuration is stored."""
        config = A2AAuthConfig()
        config.add_bearer_token("test-token")

        assert config.bearer_token is not None
        assert config.bearer_token.token == "test-token"
        assert config.bearer_token.token_type == "Bearer"


class TestAPIKeyAuth:
    """Test API key authentication."""

    def test_api_key_default_header(self):
        """API key uses default X-API-Key header."""
        config = A2AAuthConfig()
        config.add_api_key("secret-key")

        headers = config.build_headers()

        assert headers["X-API-Key"] == "secret-key"

    def test_api_key_custom_header(self):
        """API key with custom header name."""
        config = A2AAuthConfig()
        config.add_api_key("secret-key", header_name="Authorization")

        headers = config.build_headers()

        assert headers["Authorization"] == "secret-key"

    def test_api_key_persistence(self):
        """API key configuration is stored."""
        config = A2AAuthConfig()
        config.add_api_key("my-key")

        assert config.api_key is not None
        assert config.api_key.key == "my-key"
        assert config.api_key.header_name == "X-API-Key"


class TestBasicAuth:
    """Test HTTP Basic authentication."""

    def test_basic_auth_header(self):
        """Basic auth generates correct Authorization header."""
        config = A2AAuthConfig()
        config.add_basic_auth("user", "pass")

        headers = config.build_headers()

        assert "Authorization" in headers
        # Verify it's properly Base64 encoded
        auth_header = headers["Authorization"]
        assert auth_header.startswith("Basic ")
        decoded = base64.b64decode(auth_header[6:]).decode()
        assert decoded == "user:pass"

    def test_basic_auth_with_special_chars(self):
        """Basic auth handles special characters."""
        config = A2AAuthConfig()
        config.add_basic_auth("user@example.com", "p@ss:word")

        headers = config.build_headers()

        auth_header = headers["Authorization"]
        decoded = base64.b64decode(auth_header[6:]).decode()
        assert decoded == "user@example.com:p@ss:word"

    def test_basic_auth_persistence(self):
        """Basic auth configuration is stored."""
        config = A2AAuthConfig()
        config.add_basic_auth("user", "password")

        assert config.basic_auth is not None
        assert config.basic_auth.username == "user"
        assert config.basic_auth.password == "password"


class TestTLSAuth:
    """Test mTLS (mutual TLS) authentication."""

    def test_tls_certificates_persistence(self):
        """TLS certificate paths are stored."""
        config = A2AAuthConfig()
        config.add_tls_certificates(
            client_cert="/path/to/cert.crt",
            client_key="/path/to/key.key",
            ca_cert="/path/to/ca.crt",
        )

        assert config.tls_certificates is not None
        assert config.tls_certificates.client_cert_path == "/path/to/cert.crt"
        assert config.tls_certificates.client_key_path == "/path/to/key.key"
        assert config.tls_certificates.ca_cert_path == "/path/to/ca.crt"

    def test_tls_without_ca_cert(self):
        """TLS configuration optional CA cert."""
        config = A2AAuthConfig()
        config.add_tls_certificates(
            client_cert="/path/to/cert.crt",
            client_key="/path/to/key.key",
        )

        assert config.tls_certificates is not None
        assert config.tls_certificates.ca_cert_path is None

    def test_tls_path_objects(self):
        """TLS accepts Path objects."""
        cert_path = Path("/path/to/cert.crt")
        key_path = Path("/path/to/key.key")

        config = A2AAuthConfig()
        config.add_tls_certificates(cert_path, key_path)

        assert config.tls_certificates is not None
        assert config.tls_certificates.client_cert_path == str(cert_path)
        assert config.tls_certificates.client_key_path == str(key_path)

    def test_build_ssl_context_no_tls(self):
        """build_ssl_context() returns None without TLS config."""
        config = A2AAuthConfig()

        assert config.build_ssl_context() is None

    @pytest.mark.skipif(
        not hasattr(ssl, "create_default_context"),
        reason="ssl.create_default_context not available",
    )
    def test_build_ssl_context_with_ca(self):
        """build_ssl_context() creates context with CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create dummy certificate files
            cert_file = tmppath / "cert.crt"
            key_file = tmppath / "key.key"
            ca_file = tmppath / "ca.crt"

            cert_file.write_text("-----BEGIN CERTIFICATE-----\nMOCK\n")
            key_file.write_text("-----BEGIN PRIVATE KEY-----\nMOCK\n")
            ca_file.write_text("-----BEGIN CERTIFICATE-----\nMOCK\n")

            config = A2AAuthConfig()
            config.add_tls_certificates(str(cert_file), str(key_file), str(ca_file))

            # Note: This will fail when actually trying to load the
            # certificate, but we can verify the context is attempted
            with pytest.raises(ssl.SSLError):
                config.build_ssl_context()


class TestCustomHeaders:
    """Test custom HTTP headers."""

    def test_custom_header(self):
        """Custom headers can be added."""
        config = A2AAuthConfig()
        config.add_custom_header("X-Custom", "value")

        headers = config.build_headers()

        assert headers["X-Custom"] == "value"

    def test_multiple_custom_headers(self):
        """Multiple custom headers can be added."""
        config = A2AAuthConfig()
        config.add_custom_header("X-Custom-1", "value1")
        config.add_custom_header("X-Custom-2", "value2")

        headers = config.build_headers()

        assert headers["X-Custom-1"] == "value1"
        assert headers["X-Custom-2"] == "value2"


class TestAuthChaining:
    """Test fluent builder pattern."""

    def test_auth_method_chaining(self):
        """Auth config supports method chaining."""
        config = (
            A2AAuthConfig()
            .add_bearer_token("token123")
            .add_custom_header("X-Custom", "value")
        )

        assert config.bearer_token is not None
        assert "X-Custom" in config._extra_headers

    def test_multiple_auth_methods(self):
        """Multiple auth methods can be configured."""
        config = (
            A2AAuthConfig()
            .add_bearer_token("token")
            .add_api_key("key")
            .add_custom_header("X-Custom", "value")
        )

        headers = config.build_headers()

        # When multiple auth methods exist, all headers are included
        # Authorization will be overwritten by the last one added
        assert config.bearer_token is not None
        assert config.api_key is not None
        assert "X-Custom" in headers

    def test_chaining_returns_self(self):
        """Chaining methods return the config object."""
        config = A2AAuthConfig()
        result = config.add_bearer_token("token")

        assert result is config


class TestClientWrapperAuth:
    """Test A2AClientWrapper auth integration."""

    def test_client_wrapper_with_auth(self):
        """A2AClientWrapper accepts auth config."""
        auth = A2AAuthConfig().add_bearer_token("token123")

        wrapper = A2AClientWrapper(
            "http://test:8080",
            auth=auth,
        )

        assert wrapper._auth is auth
        assert "Authorization" in wrapper._headers

    def test_client_wrapper_merges_headers(self):
        """Client wrapper merges auth and custom headers."""
        auth = A2AAuthConfig().add_bearer_token("token123")
        custom_headers = {"X-Custom": "value"}

        wrapper = A2AClientWrapper(
            "http://test:8080",
            headers=custom_headers,
            auth=auth,
        )

        assert wrapper._headers["Authorization"] == "Bearer token123"
        assert wrapper._headers["X-Custom"] == "value"

    def test_client_wrapper_header_precedence(self):
        """Auth headers are added to wrapper headers."""
        auth = A2AAuthConfig().add_api_key("key123", header_name="X-API-Key")
        custom_headers = {"X-API-Key": "custom"}  # Should be overwritten

        wrapper = A2AClientWrapper(
            "http://test:8080",
            headers=custom_headers,
            auth=auth,
        )

        # Auth config's headers override custom headers
        assert wrapper._headers["X-API-Key"] == "key123"


class TestRunnableWithAuth:
    """Test A2ARunnable auth integration."""

    @pytest.mark.asyncio
    async def test_runnable_from_agent_url_with_auth(self):
        """A2ARunnable.from_agent_url() accepts auth."""
        auth = A2AAuthConfig().add_bearer_token("test-token")

        with patch.object(
            A2AClientWrapper, "_ensure_client", new_callable=AsyncMock
        ) as mock_ensure:
            mock_client = AsyncMock()
            mock_ensure.return_value = mock_client

            from a2a.types import AgentCapabilities, AgentCard

            agent_card = AgentCard(
                name="test-agent",
                url="http://test:8080",
                version="1.0.0",
                description="test",
                capabilities=AgentCapabilities(),
                default_input_modes=["text/plain"],
                default_output_modes=["text/plain"],
                skills=[],
            )
            mock_client.get_agent_card = AsyncMock(return_value=agent_card)

            with patch(
                "a2a_langchain_adapters.runnable.A2AClientWrapper"
            ) as MockWrapper:
                wrapper_instance = AsyncMock(spec=A2AClientWrapper)
                wrapper_instance.agent_card = agent_card
                wrapper_instance.get_agent_card = AsyncMock(return_value=agent_card)
                wrapper_instance.requires_mTLS = MagicMock(return_value=False)
                MockWrapper.return_value = wrapper_instance

                await A2ARunnable.from_agent_url(
                    "http://test:8080",
                    auth=auth,
                )

                # Verify auth was passed to client wrapper
                MockWrapper.assert_called_once()
                call_kwargs = MockWrapper.call_args.kwargs
                assert call_kwargs["auth"] is auth

    @pytest.mark.asyncio
    async def test_runnable_requires_mtls_logging(self):
        """A2ARunnable logs when agent requires mTLS."""
        auth = A2AAuthConfig().add_tls_certificates("/path/to/cert", "/path/to/key")

        from a2a.types import AgentCapabilities, AgentCard

        agent_card = AgentCard(
            name="test-agent",
            url="https://test:8443",
            version="1.0.0",
            description="test",
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[],
        )

        with patch("a2a_langchain_adapters.runnable.A2AClientWrapper") as MockWrapper:
            wrapper_instance = MagicMock(spec=A2AClientWrapper)
            wrapper_instance.agent_card = agent_card
            wrapper_instance.get_agent_card = AsyncMock(return_value=agent_card)
            wrapper_instance.requires_mTLS = MagicMock(return_value=True)
            MockWrapper.return_value = wrapper_instance

            with patch("a2a_langchain_adapters.runnable.logger") as mock_logger:
                await A2ARunnable.from_agent_url(
                    "https://test:8443",
                    auth=auth,
                )

                # Verify logging was called
                mock_logger.info.assert_called()


class TestSecurityDiscovery:
    """Test agent card security scheme discovery."""

    def test_client_requires_mtls_no_card(self):
        """requires_mTLS() returns False without agent card."""
        wrapper = A2AClientWrapper("http://test:8080")

        assert wrapper.requires_mTLS() is False

    def test_client_requires_mtls_no_schemes(self):
        """requires_mTLS() returns False without security schemes."""
        from a2a.types import AgentCapabilities, AgentCard

        wrapper = A2AClientWrapper("http://test:8080")
        wrapper._agent_card = AgentCard(
            name="test",
            url="http://test:8080",
            version="1.0.0",
            description="test",
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[],
        )

        assert wrapper.requires_mTLS() is False

    def test_get_security_schemes_none(self):
        """get_security_schemes() returns None without card."""
        wrapper = A2AClientWrapper("http://test:8080")

        assert wrapper.get_security_schemes() is None


class TestCredentialDataclasses:
    """Test credential dataclass definitions."""

    def test_bearer_token_credentials(self):
        """BearerTokenCredentials dataclass."""
        cred = BearerTokenCredentials(token="test-token")

        assert cred.token == "test-token"
        assert cred.token_type == "Bearer"

    def test_api_key_credentials(self):
        """APIKeyCredentials dataclass."""
        cred = APIKeyCredentials(key="secret")

        assert cred.key == "secret"
        assert cred.header_name == "X-API-Key"

    def test_basic_auth_credentials(self):
        """BasicAuthCredentials dataclass."""
        cred = BasicAuthCredentials(username="user", password="pass")

        assert cred.username == "user"
        assert cred.password == "pass"

    def test_tls_certificates(self):
        """TLSCertificates dataclass."""
        certs = TLSCertificates(
            client_cert_path="/path/to/cert",
            client_key_path="/path/to/key",
            ca_cert_path="/path/to/ca",
        )

        assert certs.client_cert_path == "/path/to/cert"
        assert certs.client_key_path == "/path/to/key"
        assert certs.ca_cert_path == "/path/to/ca"
