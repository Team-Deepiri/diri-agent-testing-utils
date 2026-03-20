"""Tests for pytest fixtures provided by the package."""
from diri_agent_testing_utils.providers.fake_llm import FakeLLMProvider, ScriptedLLMProvider
from diri_agent_testing_utils.mocks.services import MockRedis, MockVectorStore, MockMemoryManager
from diri_agent_testing_utils.mocks.tool_registry import FakeToolRegistry
from diri_agent_testing_utils.harness.test_harness import AgentTestHarness


def test_fake_llm_fixture(fake_llm):
    assert isinstance(fake_llm, FakeLLMProvider)
    assert fake_llm.invoke("test") == "Test LLM response"


def test_scripted_llm_fixture(scripted_llm):
    llm = scripted_llm(["hello", "world"])
    assert isinstance(llm, ScriptedLLMProvider)
    assert llm.invoke("a") == "hello"
    assert llm.invoke("b") == "world"


def test_scripted_llm_cycle(scripted_llm):
    llm = scripted_llm(["a"], cycle=True)
    assert llm.invoke("1") == "a"
    assert llm.invoke("2") == "a"


def test_fake_tool_registry_fixture(fake_tool_registry):
    assert isinstance(fake_tool_registry, FakeToolRegistry)
    assert len(fake_tool_registry.list_tools()) == 0


def test_mock_redis_fixture(mock_redis):
    assert isinstance(mock_redis, MockRedis)
    mock_redis.set("k", "v")
    assert mock_redis.get("k") == "v"


def test_mock_vector_store_fixture(mock_vector_store):
    assert isinstance(mock_vector_store, MockVectorStore)


def test_mock_memory_manager_fixture(mock_memory_manager):
    assert isinstance(mock_memory_manager, MockMemoryManager)


def test_agent_test_harness_fixture(agent_test_harness):
    class DummyAgent:
        async def invoke(self, input_text, context=None, **kwargs):
            return type("R", (), {"content": "ok", "metadata": {}, "tool_calls": [], "confidence": 1.0})()

    harness = agent_test_harness(DummyAgent())
    assert isinstance(harness, AgentTestHarness)
