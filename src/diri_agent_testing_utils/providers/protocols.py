"""
Framework-agnostic Protocol definitions for LLM providers and agents.

These protocols mirror the interfaces used in typical AI agent systems
(e.g., Deepiri Cyrex) without importing from any specific framework.
"""
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider implementations.

    Mirrors the interface of OpenAIProvider / LocalLLMProvider in typical
    agent systems.
    """

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous invocation."""
        ...

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous invocation."""
        ...

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Streaming invocation yielding chunks."""
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return provider health status."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is available."""
        ...


@runtime_checkable
class AgentResponseProtocol(Protocol):
    """Protocol for agent response objects."""

    content: str
    metadata: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    confidence: float


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations.

    Mirrors BaseAgent.invoke() in typical agent systems.
    """

    async def invoke(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        use_tools: bool = True,
        **kwargs: Any,
    ) -> AgentResponseProtocol:
        """Invoke the agent with input text and optional context."""
        ...

    def register_tool(
        self, name: str, func: Callable[..., Any], description: str = ""
    ) -> None:
        """Register a tool with the agent."""
        ...
