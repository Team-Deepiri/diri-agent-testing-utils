"""
Fake LLM providers for deterministic testing.

Provides FakeLLMProvider (canned responses) and ScriptedLLMProvider
(pre-defined sequence) that conform to LLMProviderProtocol.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


class FakeLLMProvider:
    """LLM provider that returns deterministic responses.

    Args:
        default_response: Response returned when no response_map match is found.
        response_map: Dict mapping prompt substrings to responses. First match wins.
    """

    def __init__(
        self,
        default_response: str = "Test LLM response",
        response_map: Optional[Dict[str, str]] = None,
    ):
        self.default_response = default_response
        self.response_map = response_map or {}
        self.call_log: List[Dict[str, Any]] = []
        self.model = "fake-llm"

    def _resolve_response(self, prompt: str) -> str:
        for substring, response in self.response_map.items():
            if substring in prompt:
                return response
        return self.default_response

    def _log_call(self, prompt: str, method: str, **kwargs: Any) -> None:
        self.call_log.append({
            "prompt": prompt,
            "method": method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kwargs": kwargs,
        })

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Return a deterministic response based on the prompt."""
        self._log_call(prompt, "invoke", **kwargs)
        return self._resolve_response(prompt)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Async variant of invoke."""
        self._log_call(prompt, "ainvoke", **kwargs)
        return self._resolve_response(prompt)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield response one word at a time to simulate streaming."""
        self._log_call(prompt, "stream", **kwargs)
        response = self._resolve_response(prompt)
        words = response.split()
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"

    def health_check(self) -> Dict[str, Any]:
        """Always returns healthy."""
        return {"status": "healthy", "backend": "fake", "model": self.model}

    def is_available(self) -> bool:
        """Always available."""
        return True

    def reset(self) -> None:
        """Clear call log."""
        self.call_log.clear()


class ScriptedLLMProvider:
    """LLM provider that returns a pre-defined sequence of responses.

    Args:
        responses: Ordered list of responses to return.
        cycle: If True, cycle through responses when exhausted.
               If False, raise IndexError when exhausted.
    """

    def __init__(self, responses: List[str], cycle: bool = False):
        self.responses = list(responses)
        self.cycle = cycle
        self._index = 0
        self.call_log: List[Dict[str, Any]] = []
        self.model = "scripted-llm"

    def _next_response(self) -> str:
        if not self.responses:
            raise IndexError("ScriptedLLMProvider has no responses configured")
        if self._index >= len(self.responses):
            if self.cycle:
                self._index = 0
            else:
                raise IndexError(
                    f"ScriptedLLMProvider exhausted after {len(self.responses)} responses"
                )
        response = self.responses[self._index]
        self._index += 1
        return response

    def _log_call(self, prompt: str, method: str, **kwargs: Any) -> None:
        self.call_log.append({
            "prompt": prompt,
            "method": method,
            "response_index": self._index - 1 if self._index > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kwargs": kwargs,
        })

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Return the next scripted response."""
        response = self._next_response()
        self._log_call(prompt, "invoke", **kwargs)
        return response

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Async variant of invoke."""
        response = self._next_response()
        self._log_call(prompt, "ainvoke", **kwargs)
        return response

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield the next scripted response one word at a time."""
        response = self._next_response()
        self._log_call(prompt, "stream", **kwargs)
        words = response.split()
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"

    def health_check(self) -> Dict[str, Any]:
        """Always returns healthy."""
        return {
            "status": "healthy",
            "backend": "scripted",
            "model": self.model,
            "remaining": max(0, len(self.responses) - self._index),
        }

    def is_available(self) -> bool:
        """Available if there are remaining responses (or cycling)."""
        return self.cycle or self._index < len(self.responses)

    def reset(self) -> None:
        """Reset to the beginning of the script."""
        self._index = 0
        self.call_log.clear()
