"""
Pytest fixtures auto-registered via the pytest11 entry point.

When diri-agent-testing-utils is installed, these fixtures are
available in any test without explicit imports.
"""
import pytest

from ..providers.fake_llm import FakeLLMProvider, ScriptedLLMProvider
from ..mocks.services import MockRedis, MockVectorStore, MockMemoryManager
from ..mocks.tool_registry import FakeToolRegistry
from ..harness.test_harness import AgentTestHarness


@pytest.fixture
def fake_llm():
    """Provide a FakeLLMProvider that returns deterministic responses."""
    return FakeLLMProvider()


@pytest.fixture
def scripted_llm():
    """Factory fixture: scripted_llm(["resp1", "resp2"]) -> ScriptedLLMProvider."""

    def _factory(responses, cycle=False):
        return ScriptedLLMProvider(responses=responses, cycle=cycle)

    return _factory


@pytest.fixture
def fake_tool_registry():
    """Provide an empty FakeToolRegistry."""
    return FakeToolRegistry()


@pytest.fixture
def mock_redis():
    """Provide a MockRedis instance."""
    return MockRedis()


@pytest.fixture
def mock_vector_store():
    """Provide a MockVectorStore instance."""
    return MockVectorStore()


@pytest.fixture
def mock_memory_manager():
    """Provide a MockMemoryManager instance."""
    return MockMemoryManager()


@pytest.fixture
def agent_test_harness():
    """Factory fixture: agent_test_harness(agent) -> AgentTestHarness."""

    def _factory(agent):
        return AgentTestHarness(agent=agent)

    return _factory
