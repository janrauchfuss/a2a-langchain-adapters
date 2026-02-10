"""Authentication and security configuration for A2A agents.

Supports multiple authentication schemes:
- mTLS (mutual TLS with client certificates)
- Bearer tokens (JWT, OAuth2)
- API keys
- Basic authentication
"""

from __future__ import annotations

import base64
import ssl
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TLSCertificates:
    """mTLS client certificates.

    Attributes:
        client_cert_path: Path to client certificate file (.crt).
        client_key_path: Path to client private key file (.key).
        ca_cert_path: Optional path to CA certificate for verification.
    """

    client_cert_path: str
    client_key_path: str
    ca_cert_path: str | None = None


@dataclass
class BearerTokenCredentials:
    """Bearer token credentials (e.g., JWT, OAuth2 access token).

    Attributes:
        token: The bearer token string.
        token_type: Token type prefix (default: "Bearer").
    """

    token: str
    token_type: str = "Bearer"


@dataclass
class APIKeyCredentials:
    """API Key credentials.

    Attributes:
        key: The API key value.
        header_name: HTTP header name for the key (default: "X-API-Key").
    """

    key: str
    header_name: str = "X-API-Key"


@dataclass
class BasicAuthCredentials:
    """HTTP Basic authentication credentials.

    Attributes:
        username: Username for authentication.
        password: Password for authentication.
    """

    username: str
    password: str


class A2AAuthConfig:
    """Central authentication configuration for A2A agents.

    Supports multiple auth methods with a fluent builder interface:
    - mTLS (mutual TLS with client certificates)
    - Bearer tokens (JWT, OAuth2)
    - API keys
    - Basic authentication

    Example:
        ```python
        auth = (
            A2AAuthConfig()
            .add_tls_certificates(
                client_cert="path/to/cert.crt",
                client_key="path/to/key.key",
                ca_cert="path/to/ca.crt"
            )
            .add_bearer_token(token="eyJhbGc...")
        )

        agent = await A2ARunnable.from_agent_url(
            "https://agent.example.com",
            auth=auth
        )
    """

    def __init__(self) -> None:
        """Initialize empty auth configuration."""
        self.tls_certificates: TLSCertificates | None = None
        self.bearer_token: BearerTokenCredentials | None = None
        self.api_key: APIKeyCredentials | None = None
        self.basic_auth: BasicAuthCredentials | None = None
        self._extra_headers: dict[str, str] = {}

    def add_tls_certificates(
        self,
        client_cert: str | Path,
        client_key: str | Path,
        ca_cert: str | Path | None = None,
    ) -> A2AAuthConfig:
        """Add mTLS client certificates.

        Args:
            client_cert: Path to client certificate file.
            client_key: Path to client private key file.
            ca_cert: Optional path to CA certificate for verification.

        Returns:
            Self for method chaining.
        """
        self.tls_certificates = TLSCertificates(
            client_cert_path=str(client_cert),
            client_key_path=str(client_key),
            ca_cert_path=str(ca_cert) if ca_cert else None,
        )
        return self

    def add_bearer_token(
        self,
        token: str,
        token_type: str = "Bearer",
    ) -> A2AAuthConfig:
        """Add Bearer token (JWT, OAuth2).

        Args:
            token: The bearer token string.
            token_type: Token type prefix (default: "Bearer").

        Returns:
            Self for method chaining.
        """
        self.bearer_token = BearerTokenCredentials(
            token=token,
            token_type=token_type,
        )
        return self

    def add_api_key(
        self,
        key: str,
        header_name: str = "X-API-Key",
    ) -> A2AAuthConfig:
        """Add API key credentials.

        Args:
            key: The API key value.
            header_name: HTTP header name (default: "X-API-Key").

        Returns:
            Self for method chaining.
        """
        self.api_key = APIKeyCredentials(
            key=key,
            header_name=header_name,
        )
        return self

    def add_basic_auth(
        self,
        username: str,
        password: str,
    ) -> A2AAuthConfig:
        """Add HTTP Basic authentication.

        Args:
            username: Username for authentication.
            password: Password for authentication.

        Returns:
            Self for method chaining.
        """
        self.basic_auth = BasicAuthCredentials(
            username=username,
            password=password,
        )
        return self

    def add_custom_header(
        self,
        header_name: str,
        value: str,
    ) -> A2AAuthConfig:
        """Add custom HTTP header.

        Args:
            header_name: Header name.
            value: Header value.

        Returns:
            Self for method chaining.
        """
        self._extra_headers[header_name] = value
        return self

    def build_headers(self) -> dict[str, str]:
        """Build HTTP headers for configured authentication.

        Returns:
            Dictionary of HTTP headers.
        """
        headers = dict(self._extra_headers)

        if self.bearer_token:
            headers["Authorization"] = (
                f"{self.bearer_token.token_type} {self.bearer_token.token}"
            )

        if self.api_key:
            headers[self.api_key.header_name] = self.api_key.key

        if self.basic_auth:
            credentials = base64.b64encode(
                f"{self.basic_auth.username}:{self.basic_auth.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    def build_ssl_context(self) -> ssl.SSLContext | None:
        """Build SSL context for mTLS.

        Returns:
            SSL context configured for mTLS, or None if not configured.

        Raises:
            FileNotFoundError: If certificate files not found.
            ssl.SSLError: If certificate configuration is invalid.
        """
        if not self.tls_certificates:
            return None

        context = ssl.create_default_context(
            cafile=self.tls_certificates.ca_cert_path
            if self.tls_certificates.ca_cert_path
            else None
        )
        context.load_cert_chain(
            certfile=self.tls_certificates.client_cert_path,
            keyfile=self.tls_certificates.client_key_path,
        )
        return context
