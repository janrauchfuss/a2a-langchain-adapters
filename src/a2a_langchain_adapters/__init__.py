"""LangChain A2A Adapters — integrate A2A agents into LangChain/LangGraph.

Agent-first architecture: A2ARunnable is the primary abstraction,
with .as_tool() available for LLM function-calling use cases.
"""

from .auth import (
    A2AAuthConfig,
    APIKeyCredentials,
    BasicAuthCredentials,
    BearerTokenCredentials,
    TLSCertificates,
)
from .client_wrapper import A2AClientWrapper
from .exceptions import (
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
)
from .runnable import A2ARunnable
from .toolkit import A2AToolkit
from .types import A2AResult, A2AStreamEvent

__all__ = [
    "A2AAdapterError",
    "A2AAuthConfig",
    "A2AAuthRequiredError",
    "A2ACapabilityError",
    "A2AClientWrapper",
    "A2AConnectionError",
    "A2AContentTypeError",
    "A2AProtocolError",
    "A2AResult",
    "A2ARunnable",
    "A2AStreamEvent",
    "A2ATaskNotCancelableError",
    "A2ATaskNotFoundError",
    "A2ATimeoutError",
    "A2AToolkit",
    "A2AUnsupportedOperationError",
    "APIKeyCredentials",
    "BasicAuthCredentials",
    "BearerTokenCredentials",
    "TLSCertificates",
]
