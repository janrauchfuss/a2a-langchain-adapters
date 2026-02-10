# Documentation

Welcome to the **A2A LangChain Adapters** documentation. This guide will help you integrate A2A protocol agents into your LangChain/LangGraph applications with full support for streaming, multi-turn conversations, and structured data.

---

## 📖 Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [**Getting Started**](./getting-started.md) | Installation, prerequisites, and your first query | New users, quick setup |
| [**Usage Guide**](./usage.md) | Patterns, examples, and best practices | Active developers |
| [**Configuration Reference**](./configuration.md) | Auth, transport, security, and advanced options | DevOps, configuration |

---

## 🚀 Getting Started in 5 Minutes

### 1. Install the package

```bash
# Basic (HTTP only)
pip install a2a-langchain-adapters

# With gRPC support
pip install a2a-langchain-adapters[grpc]
```

### 2. Connect to your first agent

```python
import asyncio
from a2a_langchain_adapters import A2ARunnable

async def main():
    # Connect to an A2A-compliant agent
    agent = await A2ARunnable.from_agent_url("http://localhost:8080")
    
    # Send a message
    result = await agent.ainvoke("What is 2 + 2?")
    print(result.text)
    
    await agent.close()

asyncio.run(main())
```

Need more details? → **[Go to Getting Started](./getting-started.md)**

---

## 📚 Documentation Structure

### **For First-Time Users**

Start with [Getting Started](./getting-started.md) to:
- Install the package
- Understand prerequisites
- Run your first query
- Verify everything works

### **For Building Applications**

Read [Usage Guide](./usage.md) to learn:
- Streaming responses in real-time
- Managing multi-turn conversations
- Working with structured data
- Handling files (upload/download)
- Integrating with LangChain tools
- Error handling and resilience
- Best practices and patterns

### **For Production Deployments**

Check [Configuration Reference](./configuration.md) for:
- Authentication methods (Bearer tokens, API keys, mTLS)
- Transport selection (HTTP vs gRPC)
- Timeout and retry configuration
- Logging and observability
- Security (SSL/TLS, custom headers)
- Performance tuning
- Advanced options

---

## 🎯 Common Use Cases

**"I want to get started quickly"**
→ [Getting Started](./getting-started.md) → [Your First Query](#-getting-started-in-5-minutes)

**"I need to stream responses"**
→ [Usage Guide: Streaming Responses](./usage.md#streaming-responses)

**"I'm having authentication issues"**
→ [Configuration: Authentication](./configuration.md#authentication)

**"I need to handle multi-turn conversations"**
→ [Usage Guide: Multi-Turn Conversations](./usage.md#multi-turn-conversations)

**"I want to upload/download files"**
→ [Usage Guide: File Handling](./usage.md#file-handling)

**"I need to secure my connection"**
→ [Configuration: SSL/TLS](./configuration.md#ssltls)

**"I want to use LangChain tools"**
→ [Usage Guide: LangChain Integration](./usage.md#langchain-integration)

---

## 💡 Key Features at a Glance

✅ **Agent-First Architecture** — Full A2A protocol semantics  
✅ **Multi-Transport Support** — HTTP and gRPC with auto-detection  
✅ **Streaming** — Real-time Server-Sent Events  
✅ **Multi-Turn Conversations** — Stateful context management  
✅ **Structured Data** — JSON-RPC alongside text  
✅ **File Handling** — Upload and download files  
✅ **Authentication** — Bearer tokens, API keys, mTLS  
✅ **Type-Safe** — Full mypy strict type checking  

---

## 🔗 Related Resources

- **GitHub Repository**: [a2a-langchain-adapters](https://github.com/janrauchfuss/a2a-langchain-adapters)
- **A2A Protocol**: [A2A Project](https://github.com/a2aproject)
- **LangChain Docs**: [LangChain Documentation](https://python.langchain.com)
- **Sample Agent**: [A2A Sample Agent](https://github.com/a2aproject/a2a-sample-agent)

---

## 📋 Next Steps

1. **New to the library?** → Start with [Getting Started](./getting-started.md)
2. **Ready to build?** → Explore [Usage Guide](./usage.md)
3. **Need to configure?** → Check [Configuration Reference](./configuration.md)
4. **Found a problem?** → Submit an issue on [GitHub](https://github.com/janrauchfuss/a2a-langchain-adapters/issues)

---

## 📝 Document Version

This documentation is for **a2a-langchain-adapters**. See [PyPI](https://pypi.org/project/a2a-langchain-adapters/) for version history.
