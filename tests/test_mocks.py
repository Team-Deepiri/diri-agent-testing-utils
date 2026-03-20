"""Tests for mock services."""
import pytest
from diri_agent_testing_utils.mocks.services import (
    MockRedis,
    MockVectorStore,
    MockMemoryManager,
)
from diri_agent_testing_utils.mocks.tool_registry import FakeToolRegistry


class TestMockRedis:
    def test_get_set(self):
        redis = MockRedis()
        redis.set("key1", "value1")
        assert redis.get("key1") == "value1"

    def test_get_missing(self):
        redis = MockRedis()
        assert redis.get("nonexistent") is None

    def test_delete(self):
        redis = MockRedis()
        redis.set("key1", "value1")
        redis.set("key2", "value2")
        count = redis.delete("key1", "key2", "key3")
        assert count == 2
        assert redis.get("key1") is None

    def test_exists(self):
        redis = MockRedis()
        redis.set("key1", "value1")
        assert redis.exists("key1") == 1
        assert redis.exists("key1", "key2") == 1
        assert redis.exists("nonexistent") == 0

    def test_keys_all(self):
        redis = MockRedis()
        redis.set("a", 1)
        redis.set("b", 2)
        assert sorted(redis.keys()) == ["a", "b"]

    def test_keys_prefix(self):
        redis = MockRedis()
        redis.set("user:1", "alice")
        redis.set("user:2", "bob")
        redis.set("session:1", "abc")
        assert sorted(redis.keys("user:*")) == ["user:1", "user:2"]

    def test_flushdb(self):
        redis = MockRedis()
        redis.set("key1", "value1")
        redis.flushdb()
        assert redis.get("key1") is None


class TestMockVectorStore:
    def test_add_and_search(self):
        store = MockVectorStore()
        store.add_documents([
            {"content": "Python is great", "metadata": {"lang": "en"}},
            {"content": "Java is verbose", "metadata": {"lang": "en"}},
        ])
        results = store.similarity_search("Python", k=1)
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_search_returns_all_when_no_match(self):
        store = MockVectorStore()
        store.add_documents([{"content": "doc1"}, {"content": "doc2"}])
        results = store.similarity_search("nonexistent", k=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_async_search(self):
        store = MockVectorStore()
        store.add_documents([{"content": "test doc"}])
        results = await store.asimilarity_search("test")
        assert len(results) == 1

    def test_stats(self):
        store = MockVectorStore()
        store.add_documents([{"content": "doc1"}])
        stats = store.stats()
        assert stats["num_entities"] == 1

    def test_reset(self):
        store = MockVectorStore()
        store.add_documents([{"content": "doc1"}])
        store.reset()
        assert store.stats()["num_entities"] == 0


class TestMockMemoryManager:
    @pytest.mark.asyncio
    async def test_store_and_search(self):
        mm = MockMemoryManager()
        await mm.store_memory("user asked about weather", memory_type="episodic")
        await mm.store_memory("user prefers dark mode", memory_type="semantic")

        results = await mm.search_memories("weather")
        assert len(results) == 1
        assert "weather" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_by_type(self):
        mm = MockMemoryManager()
        await mm.store_memory("episodic fact", memory_type="episodic")
        await mm.store_memory("semantic fact", memory_type="semantic")

        results = await mm.search_memories("fact", memory_type="semantic")
        assert len(results) == 1
        assert results[0]["memory_type"] == "semantic"

    @pytest.mark.asyncio
    async def test_build_context(self):
        mm = MockMemoryManager()
        await mm.store_memory("relevant context here", session_id="s1")
        ctx = await mm.build_context("relevant", session_id="s1")
        assert ctx["count"] == 1
        assert ctx["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_reset(self):
        mm = MockMemoryManager()
        await mm.store_memory("data")
        mm.reset()
        results = await mm.search_memories("data")
        assert len(results) == 0


class TestFakeToolRegistry:
    def test_register_and_list(self):
        reg = FakeToolRegistry()
        reg.register_tool("greet", lambda name: f"Hello {name}", "Greet someone")
        assert "greet" in reg.list_tools()

    def test_get_tool(self):
        reg = FakeToolRegistry()
        fn = lambda: "result"
        reg.register_tool("test_tool", fn)
        assert reg.get_tool("test_tool") is fn
        assert reg.get_tool("nonexistent") is None

    def test_execute(self):
        reg = FakeToolRegistry()
        reg.register_tool("add", lambda a, b: a + b, "Add two numbers")
        result = reg.execute("add", a=2, b=3)
        assert result == 5
        assert len(reg.call_log) == 1
        assert reg.call_log[0]["tool_name"] == "add"
        assert reg.call_log[0]["result"] == 5

    def test_execute_missing_tool(self):
        reg = FakeToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.execute("nonexistent")

    def test_get_calls_for(self):
        reg = FakeToolRegistry()
        reg.register_tool("a", lambda: 1)
        reg.register_tool("b", lambda: 2)
        reg.execute("a")
        reg.execute("b")
        reg.execute("a")
        assert len(reg.get_calls_for("a")) == 2
        assert len(reg.get_calls_for("b")) == 1

    def test_get_tool_descriptions(self):
        reg = FakeToolRegistry()
        reg.register_tool("tool1", lambda: None, "Description 1")
        descs = reg.get_tool_descriptions()
        assert descs["tool1"] == "Description 1"

    def test_reset(self):
        reg = FakeToolRegistry()
        reg.register_tool("test", lambda: None)
        reg.execute("test")
        reg.reset()
        assert len(reg.list_tools()) == 0
        assert len(reg.call_log) == 0
