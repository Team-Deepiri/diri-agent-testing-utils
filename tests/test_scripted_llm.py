"""Tests for ScriptedLLMProvider."""
import pytest
from diri_agent_testing_utils.providers.fake_llm import ScriptedLLMProvider


class TestScriptedLLMProvider:
    def test_sequential_responses(self):
        llm = ScriptedLLMProvider(responses=["first", "second", "third"])
        assert llm.invoke("a") == "first"
        assert llm.invoke("b") == "second"
        assert llm.invoke("c") == "third"

    def test_exhaustion_raises(self):
        llm = ScriptedLLMProvider(responses=["only one"])
        llm.invoke("a")
        with pytest.raises(IndexError, match="exhausted"):
            llm.invoke("b")

    def test_cycle_mode(self):
        llm = ScriptedLLMProvider(responses=["a", "b"], cycle=True)
        assert llm.invoke("1") == "a"
        assert llm.invoke("2") == "b"
        assert llm.invoke("3") == "a"
        assert llm.invoke("4") == "b"

    def test_empty_responses_raises(self):
        llm = ScriptedLLMProvider(responses=[])
        with pytest.raises(IndexError, match="no responses"):
            llm.invoke("test")

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        llm = ScriptedLLMProvider(responses=["async first", "async second"])
        assert await llm.ainvoke("a") == "async first"
        assert await llm.ainvoke("b") == "async second"

    def test_stream(self):
        llm = ScriptedLLMProvider(responses=["hello world"])
        chunks = list(llm.stream("test"))
        assert "".join(chunks) == "hello world"

    def test_call_log(self):
        llm = ScriptedLLMProvider(responses=["r1", "r2"])
        llm.invoke("prompt1")
        llm.invoke("prompt2")
        assert len(llm.call_log) == 2
        assert llm.call_log[0]["response_index"] == 0
        assert llm.call_log[1]["response_index"] == 1

    def test_health_check_remaining(self):
        llm = ScriptedLLMProvider(responses=["a", "b", "c"])
        assert llm.health_check()["remaining"] == 3
        llm.invoke("x")
        assert llm.health_check()["remaining"] == 2

    def test_is_available(self):
        llm = ScriptedLLMProvider(responses=["a"])
        assert llm.is_available() is True
        llm.invoke("x")
        assert llm.is_available() is False

    def test_is_available_cycle(self):
        llm = ScriptedLLMProvider(responses=["a"], cycle=True)
        llm.invoke("x")
        assert llm.is_available() is True

    def test_reset(self):
        llm = ScriptedLLMProvider(responses=["a", "b"])
        llm.invoke("x")
        llm.reset()
        assert llm.invoke("y") == "a"
        assert len(llm.call_log) == 1
