"""Tests for AgentTestHarness."""
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from diri_agent_testing_utils.harness import (
    AgentTestHarness,
    ConversationScript,
    ConversationStep,
    InvocationTrace,
)
from diri_agent_testing_utils.types import FakeAgentResponse


class MockAgent:
    """Minimal agent for testing the harness."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._index = 0
        self.received_inputs: List[str] = []

    async def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FakeAgentResponse:
        self.received_inputs.append(input_text)
        if self._index < len(self._responses):
            content = self._responses[self._index]
            self._index += 1
        else:
            content = "No more responses"

        return FakeAgentResponse(
            content=content,
            metadata={"step": self._index},
            tool_calls=[],
            confidence=0.9,
        )


class MockAgentWithTools:
    """Agent that reports tool calls."""

    async def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FakeAgentResponse:
        return FakeAgentResponse(
            content="Used a tool",
            tool_calls=[{"name": "search", "args": {"q": input_text}}],
            confidence=0.95,
        )

    def register_tool(self, name, func, description=""):
        pass


class ErrorAgent:
    """Agent that raises an error."""

    async def invoke(self, input_text, context=None, **kwargs):
        raise ValueError("Agent failed")


class TestAgentTestHarness:
    @pytest.mark.asyncio
    async def test_step(self):
        agent = MockAgent(["Hello back!"])
        harness = AgentTestHarness(agent)
        trace = await harness.step("Hello!")
        assert trace.response_content == "Hello back!"
        assert trace.input_text == "Hello!"
        assert trace.confidence == 0.9
        assert trace.duration_ms > 0

    @pytest.mark.asyncio
    async def test_traces_accumulated(self):
        agent = MockAgent(["r1", "r2", "r3"])
        harness = AgentTestHarness(agent)
        await harness.step("a")
        await harness.step("b")
        await harness.step("c")
        assert len(harness.traces) == 3

    @pytest.mark.asyncio
    async def test_run_conversation(self):
        agent = MockAgent(["First reply", "Second reply"])
        harness = AgentTestHarness(agent)
        script = ConversationScript(
            name="test_script",
            steps=[
                ConversationStep(input_text="msg1"),
                ConversationStep(input_text="msg2"),
            ],
        )
        traces = await harness.run_conversation(script)
        assert len(traces) == 2
        assert traces[0].response_content == "First reply"
        assert traces[1].response_content == "Second reply"

    @pytest.mark.asyncio
    async def test_run_conversation_with_expectations(self):
        agent = MockAgent(["Hello there friend"])
        harness = AgentTestHarness(agent)
        script = ConversationScript(
            steps=[
                ConversationStep(
                    input_text="Hi",
                    expected_contains=["hello"],
                    min_confidence=0.5,
                ),
            ],
        )
        traces = await harness.run_conversation(script)
        assert len(traces) == 1

    @pytest.mark.asyncio
    async def test_assert_response_contains(self):
        trace = InvocationTrace(
            input_text="test",
            response_content="The weather is sunny today",
        )
        AgentTestHarness.assert_response_contains(trace, "sunny")
        with pytest.raises(AssertionError, match="Expected response to contain"):
            AgentTestHarness.assert_response_contains(trace, "rainy")

    @pytest.mark.asyncio
    async def test_assert_tool_called(self):
        agent = MockAgentWithTools()
        harness = AgentTestHarness(agent)
        trace = await harness.step("search something")
        harness.assert_tool_called(trace, "search")
        with pytest.raises(AssertionError, match="Expected tool"):
            harness.assert_tool_called(trace, "nonexistent_tool")

    @pytest.mark.asyncio
    async def test_assert_confidence_above(self):
        trace = InvocationTrace(input_text="test", confidence=0.85)
        AgentTestHarness.assert_confidence_above(trace, 0.8)
        with pytest.raises(AssertionError, match="Expected confidence"):
            AgentTestHarness.assert_confidence_above(trace, 0.9)

    @pytest.mark.asyncio
    async def test_assert_no_error(self):
        trace = InvocationTrace(input_text="test")
        AgentTestHarness.assert_no_error(trace)
        trace_with_error = InvocationTrace(input_text="test", error="boom")
        with pytest.raises(AssertionError, match="Expected no error"):
            AgentTestHarness.assert_no_error(trace_with_error)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        agent = ErrorAgent()
        harness = AgentTestHarness(agent)
        trace = await harness.step("trigger error")
        assert trace.error == "Agent failed"
        assert trace.response_content == ""

    @pytest.mark.asyncio
    async def test_get_all_tool_calls(self):
        agent = MockAgentWithTools()
        harness = AgentTestHarness(agent)
        await harness.step("q1")
        await harness.step("q2")
        all_calls = harness.get_all_tool_calls()
        assert len(all_calls) == 2

    @pytest.mark.asyncio
    async def test_reset(self):
        agent = MockAgent(["r1"])
        harness = AgentTestHarness(agent)
        await harness.step("msg")
        assert len(harness.traces) == 1
        harness.reset()
        assert len(harness.traces) == 0
