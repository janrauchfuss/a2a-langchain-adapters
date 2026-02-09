# Getting Started with LangChain A2A Adapters

Get up and running with A2A agent integration in minutes.

## Prerequisites

- **Python 3.13+** (check with `python --version`)
- **pip** or **uv** for package management
- An **A2A-compliant agent** running locally or remotely
  - For testing: [A2A Sample Agent](https://github.com/a2aproject/a2a-sample-agent) or use our test fixtures

## Installation

Choose based on your transport needs:

**HTTP only (recommended for most users):**
```bash
pip install langchain-a2a-adapters
```

**With gRPC support:**
```bash
pip install langchain-a2a-adapters[grpc]
```

**Development mode:**
```bash
git clone https://github.com/janrauchfuss/langchain-a2a-adapters.git
cd langchain-a2a-adapters
pip install -e ".[grpc]"
```

**Verify installation:**
```python
import langchain_a2a_adapters
print(langchain_a2a_adapters.__version__)
```

## Your First Query

The simplest way to use an A2A agent:

```python
import asyncio
from langchain_a2a_adapters import A2ARunnable

async def main():
    # Create a runnable by discovering the agent
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Send a query
    result = await agent.ainvoke("What is 2 + 2?")
    
    # Access the response
    print(f"Answer: {result.text}")
    
    # Clean up
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
```

That's it! What happens:
1. `A2ARunnable.from_agent_url()` discovers the agent's capabilities
2. `ainvoke()` sends your input to the agent
3. `result.text` contains the response
4. `close()` cleans up the connection

## What's Next?

- **[Usage Guide](./usage.md)** — Streaming, multi-turn conversations, structured data, file handling, LangChain integration, error handling
- **[Configuration](./configuration.md)** — Authentication, transport selection, timeouts, logging
- **[Concepts](./concept-langchain-a2a-adapters.md)** — Deep dive into design and architecture
- **[A2A Protocol Spec](https://github.com/a2aproject/A2A)** — Official specification
