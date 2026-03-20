"""
Agent test harness for scripted conversation testing.
"""
import time
from typing import Any, Dict, List, Optional

from .conversation import ConversationScript, ConversationStep
from .trace import InvocationTrace


class AgentTestHarness:
    """Test harness for stepping an agent through scripted conversations.

    Works with any agent that has an async ``invoke(input_text, context)``
    method returning an object with ``content``, ``metadata``,
    ``tool_calls``, and ``confidence`` attributes.

    Example::

        harness = AgentTestHarness(my_agent)
        trace = await harness.step("Hello!")
        harness.assert_response_contains(trace, "hello")
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.traces: List[InvocationTrace] = []

    async def step(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InvocationTrace:
        """Send a single input to the agent and record the trace."""
        ctx = context or {}
        start = time.perf_counter()
        error = ""

        try:
            response = await self.agent.invoke(input_text, context=ctx)
            content = getattr(response, "content", str(response))
            metadata = getattr(response, "metadata", {})
            tool_calls = getattr(response, "tool_calls", [])
            confidence = getattr(response, "confidence", 0.0)
        except Exception as exc:
            content = ""
            metadata = {}
            tool_calls = []
            confidence = 0.0
            error = str(exc)

        duration_ms = (time.perf_counter() - start) * 1000

        trace = InvocationTrace(
            input_text=input_text,
            context=ctx,
            response_content=content,
            response_metadata=metadata,
            tool_calls=tool_calls,
            confidence=confidence,
            duration_ms=duration_ms,
            error=error,
        )
        self.traces.append(trace)
        return trace

    async def run_conversation(
        self, script: ConversationScript
    ) -> List[InvocationTrace]:
        """Run all steps in a conversation script and return traces.

        Also validates per-step expectations if defined.
        """
        traces: List[InvocationTrace] = []
        for step in script.steps:
            trace = await self.step(step.input_text, step.context)
            traces.append(trace)

            # Auto-validate step expectations
            if step.expected_contains:
                for substring in step.expected_contains:
                    self.assert_response_contains(trace, substring)
            if step.expected_tool_calls:
                for tool_name in step.expected_tool_calls:
                    self.assert_tool_called(trace, tool_name)
            if step.min_confidence is not None:
                self.assert_confidence_above(trace, step.min_confidence)

        return traces

    # ── Assertion helpers ──

    @staticmethod
    def assert_response_contains(trace: InvocationTrace, substring: str) -> None:
        """Assert that the response content contains a substring."""
        assert substring.lower() in trace.response_content.lower(), (
            f"Expected response to contain '{substring}', "
            f"got: '{trace.response_content[:200]}'"
        )

    @staticmethod
    def assert_tool_called(trace: InvocationTrace, tool_name: str) -> None:
        """Assert that a specific tool was called during the invocation."""
        called_tools = [
            tc.get("name", tc.get("tool_name", ""))
            for tc in trace.tool_calls
        ]
        assert tool_name in called_tools, (
            f"Expected tool '{tool_name}' to be called, "
            f"got: {called_tools}"
        )

    @staticmethod
    def assert_confidence_above(
        trace: InvocationTrace, threshold: float
    ) -> None:
        """Assert that the confidence score is above a threshold."""
        assert trace.confidence >= threshold, (
            f"Expected confidence >= {threshold}, got {trace.confidence}"
        )

    @staticmethod
    def assert_no_error(trace: InvocationTrace) -> None:
        """Assert that no error occurred during invocation."""
        assert not trace.error, f"Expected no error, got: '{trace.error}'"

    def get_all_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls across all traces."""
        calls: List[Dict[str, Any]] = []
        for trace in self.traces:
            calls.extend(trace.tool_calls)
        return calls

    def reset(self) -> None:
        """Clear all recorded traces."""
        self.traces.clear()
