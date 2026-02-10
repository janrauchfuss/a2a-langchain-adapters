# Configuration Guide

Complete reference for configuring `a2a-langchain-adapters` for your environment.

## Table of Contents

1. [Authentication](#authentication)
2. [Transport Selection](#transport-selection)
3. [Timeouts](#timeouts)
4. [Logging](#logging)
5. [Custom Headers](#custom-headers)
6. [SSL/TLS](#ssltls)
7. [Performance Tuning](#performance-tuning)
8. [Advanced Configuration](#advanced-configuration)

## Authentication

A2A agents often require authentication. The library supports multiple auth schemes with a fluent builder interface.

### Bearer Token (JWT / OAuth2)

Use bearer tokens for OAuth2 and JWT-based authentication:

```python
from a2a_langchain_adapters import A2ARunnable, A2AAuthConfig

auth = A2AAuthConfig().add_bearer_token(
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    token_type="Bearer"  # optional, default is "Bearer"
)

agent = await A2ARunnable.from_agent_url(
    "https://secure-agent.example.com",
    auth=auth
)
```

**Token Types:**
- `"Bearer"` — Standard OAuth2 bearer tokens (default)
- `"DPoP"` — Demonstration of Proof-of-Possession
- Custom prefix: pass any string as `token_type`

### API Key

Use API key headers for key-based authentication:

```python
auth = A2AAuthConfig().add_api_key(
    key="sk_live_1234567890abcdef",
    header_name="X-API-Key"  # optional, default is "X-API-Key"
)
```

**Common Header Names:**
- `"X-API-Key"` — Standard API key header (default)
- `"Authorization"` — Some services use this
- Custom: specify your service's expected header name

### Basic Authentication

Use username and password for HTTP Basic auth:

```python
auth = A2AAuthConfig().add_basic_auth(
    username="user@example.com",
    password="secure_password"
)
```

The library automatically encodes credentials as base64:
```
Authorization: Basic dXNlckBleGFtcGxlLmNvbTpzZWN1cmVfcGFzc3dvcmQ=
```

### mTLS (Mutual TLS)

Use client certificates for mutual TLS authentication:

```python
auth = A2AAuthConfig().add_tls_certificates(
    client_cert="/path/to/client.crt",
    client_key="/path/to/client.key",
    ca_cert="/path/to/ca.crt"  # optional
)
```

**Certificate Requirements:**
- `client_cert` — Client certificate file (.crt or .pem)
- `client_key` — Client private key file (.key or .pem)
- `ca_cert` — Optional CA certificate for custom CAs

**Example with Kubernetes certificates:**
```python
from pathlib import Path

cert_dir = Path("/var/run/secrets/kubernetes.io/serviceaccount")

auth = A2AAuthConfig().add_tls_certificates(
    client_cert=cert_dir / "tls.crt",
    client_key=cert_dir / "tls.key",
    ca_cert=cert_dir / "ca.crt"
)
```

### Chaining Multiple Auth Methods

Combine multiple auth schemes:

```python
auth = (
    A2AAuthConfig()
    .add_tls_certificates(
        client_cert="/path/to/cert.crt",
        client_key="/path/to/key.key"
    )
    .add_bearer_token(token="eyJ...")
    .add_custom_header("X-Client-ID", "my-app-v1")
    .add_custom_header("X-Request-ID", str(uuid4()))
)
```

The library sends all configured credentials with each request.

### Custom Headers

Add arbitrary HTTP headers for specialized auth schemes:

```python
auth = (
    A2AAuthConfig()
    .add_custom_header("X-API-Version", "2024-01")
    .add_custom_header("X-Client-ID", "my-service")
    .add_custom_header("X-Correlation-ID", correlation_id)
)
```

### Loading Credentials from Environment Variables

```python
import os
from a2a_langchain_adapters import A2AAuthConfig

# From environment variables
api_key = os.getenv("A2A_API_KEY")
token = os.getenv("A2A_BEARER_TOKEN")
username = os.getenv("A2A_USERNAME")
password = os.getenv("A2A_PASSWORD")

auth = A2AAuthConfig()

if api_key:
    auth.add_api_key(api_key)

if token:
    auth.add_bearer_token(token)

if username and password:
    auth.add_basic_auth(username, password)

# Use with agent
agent = await A2ARunnable.from_agent_url(
    "https://agent.example.com",
    auth=auth
)
```

### Secrets Management

For production, use secure secrets management:

```python
import os
from pathlib import Path
from a2a_langchain_adapters import A2AAuthConfig

# Load token from secure file (e.g., from secrets volume)
def load_token_from_secret() -> str:
    secret_path = Path("/run/secrets/a2a_token")
    if secret_path.exists():
        return secret_path.read_text().strip()
    return os.getenv("A2A_TOKEN", "")

token = load_token_from_secret()
auth = A2AAuthConfig().add_bearer_token(token)
```

## Transport Selection

Choose between HTTP and gRPC transports:

### HTTP (Default)

HTTP is the default and most compatible transport:

```python
from a2a_langchain_adapters import A2ARunnable

# Explicit HTTP (auto-detect by default)
agent = await A2ARunnable.from_agent_url(
    "http://agent.example.com",
    transport="http"  # optional
)
```

### gRPC

Use gRPC for better performance with long-lived connections:

```python
# Requires: pip install a2a-langchain-adapters[grpc]

agent = await A2ARunnable.from_agent_url(
    "grpc://agent.example.com:50051",
    transport="grpc"
)
```

### Auto-Detection

Let the library detect the best transport:

```python
# Auto-selects based on URL scheme and agent capabilities
agent = await A2ARunnable.from_agent_url(
    "http://agent.example.com"  # Uses HTTP
)

agent = await A2ARunnable.from_agent_url(
    "grpc://agent.example.com:50051"  # Uses gRPC
)
```

**Transport Selection Logic:**
1. Explicit `transport` parameter (if provided)
2. URL scheme (`http://`, `https://`, `grpc://`)
3. Agent capabilities (if gRPC available)
4. Fallback to HTTP

### When to Use Each

| Scenario | Transport |
|----------|-----------|
| Quick queries, one-off requests | HTTP |
| Streaming or long-lived connections | gRPC |
| Development, debugging | HTTP |
| Production, high throughput | gRPC |
| Legacy systems | HTTP |
| Low-latency services | gRPC |

## Timeouts

Control how long requests can run:

### Request Timeout

Set the overall request timeout:

```python
from a2a_langchain_adapters import A2ARunnable

# Timeout for discovery and individual requests
agent = await A2ARunnable.from_agent_url(
    "http://localhost:8080",
    timeout=30.0  # seconds
)

# Per-request timeout
result = await agent.ainvoke("query", timeout=60.0)
```

### Timeout Strategies

```python
# Quick queries (default)
agent_fast = await A2ARunnable.from_agent_url(url, timeout=5.0)

# Analysis/processing tasks
agent_medium = await A2ARunnable.from_agent_url(url, timeout=60.0)

# Heavy processing, file operations
agent_slow = await A2ARunnable.from_agent_url(url, timeout=300.0)

# Ultra-long running tasks
agent_very_slow = await A2ARunnable.from_agent_url(url, timeout=3600.0)
```

### Per-Operation Timeout

```python
# Override for individual operations
result = await agent.ainvoke("quick query", timeout=5.0)
result = await agent.ainvoke("slow analysis", timeout=120.0)

async for event in agent.astream("long task", timeout=300.0):
    print(event)
```

### No Timeout (Not Recommended)

```python
# Infinite timeout (use with caution)
agent = await A2ARunnable.from_agent_url(url, timeout=0)
```

## Logging

Enable detailed logging for debugging:

### Basic Logging

```python
import logging

# Enable debug logging globally
logging.basicConfig(level=logging.DEBUG)

# Or for just the A2A adapters
logger = logging.getLogger("a2a_langchain_adapters")
logger.setLevel(logging.DEBUG)

# Now use the agent
agent = await A2ARunnable.from_agent_url("http://localhost:8080")
result = await agent.ainvoke("test")  # Logs detailed info
```

### Structured Logging

```python
import logging
import json

# JSON-formatted logging for production
logging.basicConfig(
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
    level=logging.INFO,
)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("a2a_langchain_adapters")
logger.addHandler(handler)
```

### Log Levels

```python
import logging

logger = logging.getLogger("a2a_langchain_adapters")

# Minimal logging (warnings and errors only)
logger.setLevel(logging.WARNING)

# Standard logging (info, warnings, errors)
logger.setLevel(logging.INFO)

# Detailed debugging
logger.setLevel(logging.DEBUG)
```

### What Gets Logged

At different levels:
- `DEBUG` — All operations, request/response details, timing
- `INFO` — Agent discovery, connection establishment
- `WARNING` — Timeouts, retries, degraded performance
- `ERROR` — Failures, exceptions

```python
# Example debug output
# DEBUG:a2a_langchain_adapters.client_wrapper:
#   Discovering agent at http://localhost:8080
# DEBUG:a2a_langchain_adapters.client_wrapper:
#   Agent card received: Calculator (skills: [add, multiply])
# DEBUG:a2a_langchain_adapters.client_wrapper:
#   Sending message: add (timeout=30.0s)
# DEBUG:a2a_langchain_adapters.client_wrapper:
#   Response received in 0.23s
```

## Custom Headers

Add custom HTTP headers to all requests:

```python
from a2a_langchain_adapters import A2AAuthConfig

auth = (
    A2AAuthConfig()
    .add_custom_header("User-Agent", "MyApp/1.0")
    .add_custom_header("X-Request-Source", "integration")
    .add_custom_header("X-Trace-ID", trace_id)
)

agent = await A2ARunnable.from_agent_url(
    "https://agent.example.com",
    auth=auth
)
```

### Common Custom Headers

```python
auth = (
    A2AAuthConfig()
    .add_custom_header("User-Agent", "MyApp/2.0.1")
    .add_custom_header("X-Client-ID", "client-123")
    .add_custom_header("X-Correlation-ID", correlation_id)
    .add_custom_header("X-Request-ID", request_id)
    .add_custom_header("Accept-Language", "en-US")
)
```

### Via Initialization

```python
agent = await A2ARunnable.from_agent_url(
    "https://agent.example.com",
    headers={
        "User-Agent": "MyApp/1.0",
        "X-Custom-Header": "value",
    }
)
```

## SSL/TLS

Control SSL/TLS behavior:

### Certificate Verification

By default, the library verifies SSL certificates:

```python
import ssl

# Default: verify certificates
agent = await A2ARunnable.from_agent_url(
    "https://secure-agent.example.com"
)
```

### Self-Signed Certificates

For self-signed certificates in development:

```python
from a2a_langchain_adapters import A2AAuthConfig

# Provide CA certificate for verification
auth = A2AAuthConfig().add_tls_certificates(
    client_cert="/path/to/client.crt",
    client_key="/path/to/client.key",
    ca_cert="/path/to/self-signed-ca.crt"  # Verify against custom CA
)

agent = await A2ARunnable.from_agent_url(
    "https://dev-agent.local:8443",
    auth=auth
)
```

### mTLS with Client Certificates

```python
auth = A2AAuthConfig().add_tls_certificates(
    client_cert="/path/to/client.crt",
    client_key="/path/to/client.key",
    ca_cert="/path/to/ca.crt"
)

agent = await A2ARunnable.from_agent_url(
    "https://agent.example.com",
    auth=auth
)
```

### Certificate File Formats

Supported formats:
- `.pem` — PEM text format (most common)
- `.crt` — Certificate format
- `.key` — Private key format

```bash
# Example: Creating test certificates
openssl req -new -newkey rsa:2048 -nodes -keyout client.key -out client.csr
openssl x509 -req -days 365 -in client.csr -signkey client.key -out client.crt
```

## Performance Tuning

### Connection Reuse

Reuse agents for better performance:

```python
# Good: Reuse agent
agent = await A2ARunnable.from_agent_url("http://localhost:8080")
for _ in range(100):
    result = await agent.ainvoke("query")
await agent.close()

# Suboptimal: Create new agent each time
for _ in range(100):
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    result = await agent.ainvoke("query")
    await agent.close()
```

### Concurrent Requests

Use asyncio for concurrent operations:

```python
import asyncio

async def concurrent_queries():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    queries = ["query1", "query2", "query3", ...]
    
    # Execute concurrently
    results = await asyncio.gather(
        *(agent.ainvoke(q) for q in queries),
        return_exceptions=True
    )
    
    await agent.close()
    return results
```

### Streaming for Large Responses

Use streaming instead of buffering:

```python
# Good: Stream large responses
async for event in agent.astream("large query"):
    if event.text:
        # Process incrementally
        print(event.text, end="", flush=True)

# Less efficient: Wait for full response
result = await agent.ainvoke("large query")  # Waits for completion
print(result.text)
```

### Connection Pooling

For multiple agents, use a toolkit:

```python
from a2a_langchain_adapters import A2AToolkit

# Efficient: Toolkit manages multiple connections
toolkit = await A2AToolkit.from_agent_urls([
    "http://agent1:8080",
    "http://agent2:8080",
    "http://agent3:8080",
])

results = await asyncio.gather(
    *(agent.ainvoke("query") for agent in toolkit.get_runnables())
)

await toolkit.close()
```

## Advanced Configuration

### Custom HTTP Client

For advanced HTTP customization, configure before instantiation:

```python
import httpx
from a2a_langchain_adapters import A2ARunnable

# Create custom HTTP client with limits
client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    timeout=30.0,
)

agent = await A2ARunnable.from_agent_url("http://localhost:8080")
# The agent uses the global httpx defaults
await agent.close()
```

### Debugging Agent Card Discovery

```python
import logging

logging.basicConfig(level=logging.DEBUG)

agent = await A2ARunnable.from_agent_url(
    "http://localhost:8080"
)

# Check discovered capabilities
print(f"Agent: {agent.agent_card.name}")
print(f"Skills: {[s.name for s in agent.agent_card.skills]}")
print(f"Capabilities: {agent.agent_card.capabilities}")

await agent.close()
```

### Configuration with Environment Variables

```python
import os
from pathlib import Path
from a2a_langchain_adapters import A2ARunnable, A2AAuthConfig

# Load from environment
agent_url = os.getenv("A2A_AGENT_URL", "http://localhost:8080")
timeout = float(os.getenv("A2A_TIMEOUT", "30.0"))
transport = os.getenv("A2A_TRANSPORT", None)

# Auth from environment
auth = A2AAuthConfig()
if token := os.getenv("A2A_BEARER_TOKEN"):
    auth.add_bearer_token(token)
if api_key := os.getenv("A2A_API_KEY"):
    auth.add_api_key(api_key)

# Create agent
agent = await A2ARunnable.from_agent_url(
    agent_url,
    timeout=timeout,
    transport=transport,
    auth=auth if auth.bearer_token or auth.api_key else None,
)

result = await agent.ainvoke("test")
await agent.close()
```

### Configuration File Pattern

```python
import json
from pathlib import Path
from a2a_langchain_adapters import A2ARunnable, A2AAuthConfig

config_file = Path("a2a-config.json")
config = json.loads(config_file.read_text())

auth = A2AAuthConfig()

if "bearer_token" in config:
    auth.add_bearer_token(config["bearer_token"])

if "api_key" in config:
    auth.add_api_key(
        config["api_key"]["key"],
        header_name=config["api_key"].get("header_name", "X-API-Key")
    )

agent = await A2ARunnable.from_agent_url(
    config["agent_url"],
    timeout=config.get("timeout", 30.0),
    transport=config.get("transport"),
    auth=auth,
)
```

---

## See Also

- **[Usage Guide](./usage.md)** — Practical patterns and examples
- **[Getting Started](./getting-started.md)** — Quick start guide
- **[Concepts](./concept-a2a-langchain-adapters.md)** — Architecture and design
