"""
diri-agent-testing-utils

Pytest fixtures, mocks, and test harness for AI agent and LLM testing.
"""
from .types import FakeAgentConfig, FakeAgentResponse, FakeToolCall
from .providers.fake_llm import FakeLLMProvider, ScriptedLLMProvider
from .providers.protocols import (
    LLMProviderProtocol,
    AgentProtocol,
    AgentResponseProtocol,
)
from .mocks.services import MockRedis, MockVectorStore, MockMemoryManager
from .mocks.tool_registry import FakeToolRegistry
from .harness.test_harness import AgentTestHarness
from .harness.conversation import ConversationScript, ConversationStep
from .harness.trace import InvocationTrace

__all__ = [
    # Providers
    "FakeLLMProvider",
    "ScriptedLLMProvider",
    "LLMProviderProtocol",
    "AgentProtocol",
    "AgentResponseProtocol",
    # Types
    "FakeAgentConfig",
    "FakeAgentResponse",
    "FakeToolCall",
    # Mocks
    "MockRedis",
    "MockVectorStore",
    "MockMemoryManager",
    "FakeToolRegistry",
    # Harness
    "AgentTestHarness",
    "ConversationScript",
    "ConversationStep",
    "InvocationTrace",
]
