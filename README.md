# diri-agent-testing-utils

Pytest fixtures, mocks, and test harness for AI agent and LLM testing.

## Installation

```bash
pip install diri-agent-testing-utils
```

For async support:

```bash
pip install diri-agent-testing-utils[async]
```

## Features

- **Fake LLM Providers** — Deterministic and scripted LLM providers for testing without real API calls
- **Mock Services** — In-memory mocks for Redis, vector stores, memory managers, and tool registries
- **Agent Test Harness** — Define scripted conversations, step agents through them, and assert on results
- **Pytest Fixtures** — Auto-registered fixtures via pytest plugin for quick test setup

## Quick Start

```python
from diri_agent_testing_utils.providers.fake_llm import FakeLLMProvider, ScriptedLLMProvider
from diri_agent_testing_utils.harness import AgentTestHarness, ConversationScript, ConversationStep

# Create a fake LLM that returns deterministic responses
llm = FakeLLMProvider(default_response="I can help with that!")

# Or use scripted responses
llm = ScriptedLLMProvider(responses=[
    "First response",
    "Second response",
    "Third response",
])

# Use pytest fixtures (auto-registered when package is installed)
def test_my_agent(fake_llm, fake_tool_registry):
    # fake_llm and fake_tool_registry are available automatically
    assert fake_llm.invoke("hello") == "Test LLM response"
```

## Pytest Fixtures

When installed, the following fixtures are automatically available:

| Fixture | Type | Description |
|---------|------|-------------|
| `fake_llm` | `FakeLLMProvider` | Deterministic LLM provider |
| `scripted_llm` | Factory | `scripted_llm(["resp1", "resp2"])` -> `ScriptedLLMProvider` |
| `fake_tool_registry` | `FakeToolRegistry` | In-memory tool registry with call tracking |
| `mock_redis` | `MockRedis` | In-memory Redis mock |
| `mock_vector_store` | `MockVectorStore` | In-memory vector store mock |
| `agent_test_harness` | Factory | `agent_test_harness(agent)` -> `AgentTestHarness` |

## License

MIT
