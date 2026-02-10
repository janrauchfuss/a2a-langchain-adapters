# Usage Guide

Comprehensive patterns and examples for using `a2a-langchain-adapters` in your applications.

## Table of Contents

1. [Streaming Responses](#streaming-responses)
2. [Multi-Turn Conversations](#multi-turn-conversations)
3. [Structured Data](#structured-data)
4. [File Handling](#file-handling)
5. [LangChain Integration](#langchain-integration)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Streaming Responses

For long-running or verbose tasks, stream responses in real-time:

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    print("Streaming response:")
    async for event in agent.astream("Explain quantum computing in detail"):
        # Each event is A2AStreamEvent
        if event.text:
            print(event.text, end="", flush=True)
        if event.data:
            print(f"\nStructured data received: {event.data}")
    
    await agent.close()

asyncio.run(main())
```

**Stream Events:**
- `event.text` — text chunks (concatenate for full response)
- `event.data` — structured JSON payloads
- `event.status` — task status updates
- `event.artifacts` — file uploads/downloads

### Streaming with Timeouts

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable, A2ATimeoutError

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    try:
        async for event in agent.astream("Complex analysis", timeout=60.0):
            if event.text:
                print(event.text, end="", flush=True)
    except A2ATimeoutError:
        print("\nStream timed out")
    finally:
        await agent.close()

asyncio.run(main())
```

### Collecting Streamed Data

```python
async def collect_response():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    full_text = ""
    structured_data = []
    
    async for event in agent.astream("Get analysis"):
        if event.text:
            full_text += event.text
        if event.data:
            structured_data.append(event.data)
    
    print(f"Full response: {full_text}")
    print(f"Data payloads: {structured_data}")
    
    await agent.close()

asyncio.run(collect_response())
```

## Multi-Turn Conversations

A2A agents maintain conversation context. Build stateful dialogues:

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # First turn: establish context
    result1 = await agent.ainvoke("My name is Alice. I'm interested in Python.")
    print(f"Agent: {result1.text}")
    context_id = result1.context_id
    
    # Second turn: continue in same context
    agent_conv = agent.with_context(context_id)
    result2 = await agent_conv.ainvoke("Tell me about async programming.")
    print(f"Agent: {result2.text}")
    # Agent can reference "you said you're interested in Python"
    
    # Third turn: another follow-up
    result3 = await agent_conv.ainvoke("What about type hints?")
    print(f"Agent: {result3.text}")
    
    await agent.close()

asyncio.run(main())
```

**Key points:**
- `result.context_id` captures the conversation context
- `agent.with_context(context_id)` creates a runnable in that context
- The agent remembers previous messages in the same context
- Perfect for chatbots and interview-style interactions

### Chatbot Pattern

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def chatbot_session(agent_url: str):
    """Simple chatbot with multi-turn context."""
    agent = await A2ARunnable.from_agent_url(agent_url)
    context_id = None
    
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            
            # Create runnable with context if we have one
            current_agent = (
                agent.with_context(context_id)
                if context_id
                else agent
            )
            
            result = await current_agent.ainvoke(user_input)
            print(f"Agent: {result.text}\n")
            
            # Update context for next turn
            context_id = result.context_id
    finally:
        await agent.close()

asyncio.run(chatbot_session("http://localhost:8080"))
```

### Parallel Conversations

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def parallel_conversations():
    """Maintain multiple independent conversations."""
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Start conversation 1
    conv1_first = await agent.ainvoke("I want to learn Python")
    conv1_agent = agent.with_context(conv1_first.context_id)
    
    # Start conversation 2
    conv2_first = await agent.ainvoke("I want to learn JavaScript")
    conv2_agent = agent.with_context(conv2_first.context_id)
    
    # Continue each independently
    conv1_result = await conv1_agent.ainvoke("What's the best way to start?")
    conv2_result = await conv2_agent.ainvoke("What's the best way to start?")
    
    print(f"Python advice: {conv1_result.text}")
    print(f"JavaScript advice: {conv2_result.text}")
    
    await agent.close()

asyncio.run(parallel_conversations())
```

## Structured Data

Send and receive JSON payloads, not just text:

```python
import asyncio
import json
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Send structured data
    payload = {
        "action": "analyze",
        "target": "sales_report_q4",
        "format": "summary"
    }
    
    result = await agent.ainvoke(payload)
    
    # Receive structured data
    if result.data:
        analysis = result.data
        print(f"Analysis: {json.dumps(analysis, indent=2)}")
    
    print(f"Summary: {result.text}")
    
    await agent.close()

asyncio.run(main())
```

### Mixed Input (Text + Data)

```python
async def mixed_input():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Send both text and structured data
    payload = {
        "@text": "Analyze this sales data from Q4",
        "data": {
            "q1": 100000,
            "q2": 150000,
            "q3": 200000,
            "q4": 250000
        }
    }
    
    result = await agent.ainvoke(payload)
    print(f"Analysis: {result.text}")
    if result.data:
        print(f"Metrics: {result.data}")
    
    await agent.close()

asyncio.run(mixed_input())
```

### Data Serialization

```python
from dataclasses import asdict
from typing import TypedDict

class SalesQuery(TypedDict):
    action: str
    quarters: list[int]
    region: str

async def type_safe_query():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    query: SalesQuery = {
        "action": "forecast",
        "quarters": [1, 2, 3, 4],
        "region": "EMEA"
    }
    
    result = await agent.ainvoke(query)
    print(result.text)
    
    await agent.close()

asyncio.run(type_safe_query())
```

## File Handling

Upload and download files with agents:

### File Upload

```python
import asyncio
from pathlib import Path
from a2a_langchain_adapters import A2ARunnable

async def upload_file():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Upload a file for analysis
    file_path = Path("report.pdf")
    
    result = await agent.ainvoke(
        f"Summarize {file_path.name}",
        files=[file_path]
    )
    print(f"Summary: {result.text}")
    
    await agent.close()

asyncio.run(upload_file())
```

### Multiple File Upload

```python
async def upload_multiple():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    files = [
        Path("Q1_report.pdf"),
        Path("Q2_report.pdf"),
        Path("Q3_report.pdf"),
    ]
    
    result = await agent.ainvoke(
        "Combine quarterly reports into annual summary",
        files=files
    )
    print(result.text)
    
    await agent.close()

asyncio.run(upload_multiple())
```

### File Download

```python
async def download_files():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    result = await agent.ainvoke("Generate detailed analysis report")
    
    # Download processed files (if agent provides them)
    if result.artifacts:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        for artifact in result.artifacts:
            output_path = output_dir / artifact.name
            with open(output_path, "wb") as f:
                f.write(artifact.data)
            print(f"Saved: {output_path}")
    
    await agent.close()

asyncio.run(download_files())
```

### Streaming File Processing

```python
async def stream_file_output():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    file_path = Path("large_document.pdf")
    
    async for event in agent.astream(
        f"Process {file_path.name}",
        files=[file_path]
    ):
        if event.text:
            print(event.text, end="", flush=True)
        if event.artifacts:
            for artifact in event.artifacts:
                print(f"\nGenerated: {artifact.name}")
    
    await agent.close()

asyncio.run(stream_file_output())
```

## LangChain Integration

### Expose Agent as a Tool

Integrate an A2A agent into an LLM's function-calling workflow:

```python
import asyncio
from langchain_openai import ChatOpenAI
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Convert agent to a LangChain tool
    tool = agent.as_tool()
    
    # Bind to LLM
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools([tool])
    
    # Now the LLM can call the agent
    response = llm_with_tools.invoke("Ask the assistant agent what 2+2 is")
    print(response)
    
    await agent.close()

asyncio.run(main())
```

### Multiple Agents as Tools

```python
from a2a_langchain_adapters import A2AToolkit
from langchain_openai import ChatOpenAI

async def main():
    toolkit = await A2AToolkit.from_agent_urls([
        "http://summarizer-agent:8080",
        "http://translator-agent:8080",
        "http://search-agent:8080",
    ])
    
    # Get tools for LLM function calling
    tools = toolkit.get_tools()
    
    llm = ChatOpenAI(model="gpt-4")
    llm_with_tools = llm.bind_tools(tools)
    
    # LLM can now invoke any agent
    response = llm_with_tools.invoke("Summarize the latest news")
    
    await toolkit.close()

asyncio.run(main())
```

**Note:** Tools are generated per skill. If an agent has 3 skills, `get_tools()` returns 3 tools.

### Runnable in LangChain Chains

Use A2ARunnable directly in LangChain chains:

```python
from langchain_core.runnables import RunnableSequence
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Chain: preprocess -> agent -> postprocess
    chain = (
        {
            "query": lambda x: x.upper(),  # Uppercase input
        }
        | agent  # Invoke agent
        | (lambda result: f"Final: {result.text}")  # Format output
    )
    
    result = await chain.ainvoke("hello world")
    print(result)

asyncio.run(main())
```

### In LangGraph State Graphs

```python
from langgraph.graph import StateGraph
from a2a_langchain_adapters import A2ARunnable
from typing import TypedDict

class State(TypedDict):
    user_query: str
    agent_response: str

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    def agent_node(state: State) -> State:
        # In LangGraph, you may need to handle async within sync nodes
        import asyncio
        result = asyncio.run(agent.ainvoke(state["user_query"]))
        state["agent_response"] = result.text
        return state
    
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    
    app = graph.compile()
    
    result = app.invoke({"user_query": "What is Python?"})
    print(result["agent_response"])

asyncio.run(main())
```

## Error Handling

A2A operations can fail for various reasons. Handle errors gracefully:

```python
import asyncio
from a2a_langchain_adapters import (
    A2ARunnable,
    A2AConnectionError,
    A2ATimeoutError,
    A2AAuthRequiredError,
    A2AAdapterError,
)

async def main():
    try:
        agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    except A2AConnectionError:
        print("Failed to connect to agent. Is it running?")
        return
    
    try:
        result = await agent.ainvoke("Query", timeout=10)
    except A2ATimeoutError:
        print("Agent took too long to respond")
    except A2AAuthRequiredError:
        print("Authentication failed. Check credentials.")
    except A2AAdapterError as e:
        print(f"Protocol error: {e}")
    finally:
        await agent.close()

asyncio.run(main())
```

### Robust Retry Logic

```python
import asyncio
from typing import TypeVar, Callable, Any
from a2a_langchain_adapters import A2ARunnable, A2ATimeoutError

T = TypeVar("T")

async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Retry an async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except A2ATimeoutError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Timeout, retrying in {delay}s...")
            await asyncio.sleep(delay)

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    result = await retry_with_backoff(
        lambda: agent.ainvoke("Complex query"),
        max_retries=3
    )
    print(result.text)
    
    await agent.close()

asyncio.run(main())
```

### Exception-Specific Recovery

```python
async def resilient_query():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    try:
        result = await agent.ainvoke("Query")
    except A2ATimeoutError:
        # Retry with longer timeout
        result = await agent.ainvoke("Query", timeout=60.0)
    except A2AConnectionError:
        # Try fallback agent
        agent = await A2ARunnable.from_agent_url("http://fallback-agent:8080")
        result = await agent.ainvoke("Query")
    
    return result.text
```

## Best Practices

### 1. Resource Cleanup

Always close agents when done:

```python
from a2a_langchain_adapters import A2ARunnable

async def main():
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    try:
        result = await agent.ainvoke("query")
    finally:
        await agent.close()
```

### 2. Reuse Agents

Create agents once and reuse them:

```python
async def good_example():
    # Good: One agent, reused
    agent = await A2ARunnable.from_agent_url("http://agent:8080")
    try:
        for query in queries:
            result = await agent.ainvoke(query)
    finally:
        await agent.close()

async def bad_example():
    # Bad: Creating agent repeatedly
    for query in queries:
        agent = await A2ARunnable.from_agent_url("http://agent:8080")
        result = await agent.ainvoke(query)
        await agent.close()  # Expensive!
```

### 3. Context Reuse

Leverage multi-turn context for stateful interactions:

```python
# Good: Stateful conversation
agent = await A2ARunnable.from_agent_url("http://agent:8080")
result1 = await agent.ainvoke("Set color to blue")
agent_conv = agent.with_context(result1.context_id)
result2 = await agent_conv.ainvoke("Change to red")  # Agent remembers "blue"

# Less efficient: Stateless queries
agent = await A2ARunnable.from_agent_url("http://agent:8080")
await agent.ainvoke("Set color to blue")
await agent.ainvoke("Change to red")  # Agent has no context
```

### 4. Streaming for Large Responses

Use streaming for verbose outputs to improve responsiveness:

```python
# Good: Stream long responses
async for event in agent.astream("Explain everything"):
    if event.text:
        print(event.text, end="", flush=True)

# Less user-friendly: Wait for full response
result = await agent.ainvoke("Explain everything")
print(result.text)
```

### 5. Timeout Tuning

Set appropriate timeouts for different task types:

```python
# Quick queries
agent = await A2ARunnable.from_agent_url("http://agent:8080", timeout=5.0)

# Analysis/processing
agent = await A2ARunnable.from_agent_url("http://agent:8080", timeout=60.0)

# File processing
agent = await A2ARunnable.from_agent_url("http://agent:8080", timeout=300.0)
```

### 6. Logging and Debugging

Enable logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("a2a_langchain_adapters")
logger.setLevel(logging.DEBUG)

# Now all A2A operations will be logged
agent = await A2ARunnable.from_agent_url("http://localhost:8080")
result = await agent.ainvoke("Query")  # Logged operations
```

### 7. Type Safety

Use TypedDict for structured data:

```python
from typing import TypedDict

class AnalysisRequest(TypedDict):
    action: str
    target: str
    format: str

async def typed_request():
    request: AnalysisRequest = {
        "action": "analyze",
        "target": "sales_data",
        "format": "json"
    }
    
    result = await agent.ainvoke(request)
    return result
```

### 8. Concurrent Requests

Use asyncio utilities for concurrent operations:

```python
async def concurrent_queries():
    agent = await A2ARunnable.from_agent_url("http://agent:8080")
    
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    
    # Execute concurrently
    results = await asyncio.gather(
        *(agent.ainvoke(q) for q in queries)
    )
    
    return [r.text for r in results]
```

---

## See Also

- **[Configuration](./configuration.md)** — Authentication, transport, and tuning
- **[Concepts](./concept-a2a-langchain-adapters.md)** — Architecture and design
- **[Getting Started](./getting-started.md)** — Installation and quick start
