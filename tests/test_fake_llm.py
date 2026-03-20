"""Tests for FakeLLMProvider."""
import pytest
from diri_agent_testing_utils.providers.fake_llm import FakeLLMProvider


class TestFakeLLMProvider:
    def test_default_response(self):
        llm = FakeLLMProvider(default_response="hello world")
        assert llm.invoke("anything") == "hello world"

    def test_response_map_matching(self):
        llm = FakeLLMProvider(
            default_response="default",
            response_map={
                "weather": "It's sunny",
                "time": "It's noon",
            },
        )
        assert llm.invoke("what's the weather?") == "It's sunny"
        assert llm.invoke("what time is it?") == "It's noon"
        assert llm.invoke("random question") == "default"

    def test_response_map_first_match_wins(self):
        llm = FakeLLMProvider(
            response_map={
                "hello": "first",
                "hello world": "second",
            },
        )
        assert llm.invoke("hello world") == "first"

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        llm = FakeLLMProvider(default_response="async response")
        result = await llm.ainvoke("test prompt")
        assert result == "async response"

    def test_stream(self):
        llm = FakeLLMProvider(default_response="hello beautiful world")
        chunks = list(llm.stream("test"))
        assert "".join(chunks) == "hello beautiful world"

    def test_call_log(self):
        llm = FakeLLMProvider()
        llm.invoke("first")
        llm.invoke("second")
        assert len(llm.call_log) == 2
        assert llm.call_log[0]["prompt"] == "first"
        assert llm.call_log[0]["method"] == "invoke"
        assert llm.call_log[1]["prompt"] == "second"

    @pytest.mark.asyncio
    async def test_call_log_async(self):
        llm = FakeLLMProvider()
        await llm.ainvoke("async call")
        assert len(llm.call_log) == 1
        assert llm.call_log[0]["method"] == "ainvoke"

    def test_health_check(self):
        llm = FakeLLMProvider()
        health = llm.health_check()
        assert health["status"] == "healthy"
        assert health["backend"] == "fake"

    def test_is_available(self):
        llm = FakeLLMProvider()
        assert llm.is_available() is True

    def test_reset(self):
        llm = FakeLLMProvider()
        llm.invoke("test")
        assert len(llm.call_log) == 1
        llm.reset()
        assert len(llm.call_log) == 0

    def test_kwargs_logged(self):
        llm = FakeLLMProvider()
        llm.invoke("test", temperature=0.5, max_tokens=100)
        assert llm.call_log[0]["kwargs"]["temperature"] == 0.5
        assert llm.call_log[0]["kwargs"]["max_tokens"] == 100
