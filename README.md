# A2A LangChain Adapters

A production-ready Python package for integrating A2A protocol agents into LangChain/LangGraph. Enables stateful conversations, streaming, multi-turn context management, and seamless LLM tool binding with A2A agents.

[![Build](https://github.com/janrauchfuss/langchain-a2a-adapters/actions/workflows/build_main.yml/badge.svg)](https://github.com/janrauchfuss/langchain-a2a-adapters/actions/workflows/build_main.yml)
[![DeepSource](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters.svg/?label=code+coverage&show_trend=true&token=Uc5Zf3uf_7cuQrRWlftv5UrU)](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters/)
[![DeepSource](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters.svg/?label=active+issues&show_trend=true&token=Uc5Zf3uf_7cuQrRWlftv5UrU)](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters/)
[![DeepSource](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters.svg/?label=resolved+issues&show_trend=true&token=Uc5Zf3uf_7cuQrRWlftv5UrU)](https://app.deepsource.com/gh/janrauchfuss/a2a-langchain-adapters/)
[![Ruff](https://img.shields.io/badge/ruff-enabled-success)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-checked-success)](http://mypy-lang.org/)
[![deptry](https://img.shields.io/badge/deptry-enabled-success)](https://github.com/fpgmaas/deptry)

## 🎯 Why This Package?

Integrating A2A protocol agents into LangChain/LangGraph required custom implementations each time. This package standardizes that pattern with a production-ready, well-tested solution. Currently in early release—feedback and contributions welcome.

## ✨ Features

- **Agent-First Architecture**: `A2ARunnable` preserves full A2A protocol semantics (streaming, task lifecycle, multi-turn conversations)
- **Multi-Transport Support**: HTTP (default) and gRPC with auto-detection
- **Task Resubscribe**: Reconnect to interrupted streaming tasks for resilience
- **Structured Data**: Send/receive JSON-RPC data alongside text
- **File Handling**: Upload and download files with URI and bytes patterns
- **Authentication**: Bearer tokens, API keys, mTLS, custom headers
- **LLM Tool Binding**: Expose agents as LangChain tools for function calling
- **Streaming Support**: Real-time Server-Sent Events for long-running tasks
- **Type-Safe**: Full mypy strict type checking, async-first design

## 📦 Installation

### Basic (HTTP only)

```bash
pip install a2a-langchain-adapters
```

### With gRPC Support

```bash
pip install a2a-langchain-adapters[grpc]
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def main():
    # Create agent runnable by discovering agent card
    agent = await A2ARunnable.from_agent_url("http://agent.example.com")

    # Send a message and get result
    result = await agent.ainvoke("What is 2 + 2?")
    print(f"Response: {result.text}")

    await agent.close()

asyncio.run(main())
```

### Streaming

```python
async for event in agent.astream("Explain quantum computing"):
    if event.text:
        print(event.text, end="", flush=True)
```

### Multi-Turn Conversations

```python
# Initial message
result = await agent.ainvoke("Tell me about Python")

# Follow-up in same context
agent_conv = agent.with_context(result.context_id)
followup = await agent_conv.ainvoke("What about async?")
```

### Structured Data

```python
# Send JSON data
data = {"action": "analyze", "target": "sales_q4"}
result = await agent.ainvoke(data)
print(result.data)  # Structured response
```

## 🛠️ Development

This project uses [**Just**](https://github.com/casey/just) for task automation. All commands are defined in the `Justfile`.

### Setup

```bash
just setup
```

### Available Commands

```bash
just help              # Show all available commands
just setup             # Setup development environment
just build             # Build the project
just qa                # Run quality assurance (format, lint, type check, dependency check)
just test              # Run tests
just coverage          # Run tests with coverage report
```

### Manual Commands (without Just)

```bash
# Setup
uv sync --python=3.13

# Run tests
pytest tests/ -v

# Type checking
mypy src/ --strict

# Linting and formatting
ruff format src/ tests/
ruff check --fix src/ tests/
ruff check --select I --fix src/ tests/

# Dependency checking
deptry .

# Coverage
pytest --cov=src --cov-report=html tests/
```

## ☕ Support the Project

If this project helps you, consider supporting its development:

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](buymeacoffee.com/janrauchfuss)

Your support helps keep the project maintained and growing ❤️

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Documentation

### Getting Started

- [📖 Documentation Hub](./docs/README.md) - Start here for a complete overview
- [🚀 Getting Started Guide](./docs/getting-started.md) - Installation and your first query
- [💡 Usage Guide](./docs/usage.md) - Streaming, multi-turn conversations, file handling, and best practices
- [⚙️ Configuration Reference](./docs/configuration.md) - Authentication, transport, security, and advanced options

### External Resources

- [A2A Protocol Spec](https://github.com/example/a2a-spec) - Complete protocol specification
- [LangChain Docs](https://python.langchain.com) - LangChain integration reference
