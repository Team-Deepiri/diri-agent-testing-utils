"""
Standalone type definitions for testing convenience.

These mirror typical agent system types (AgentConfig, AgentResponse, etc.)
without importing from any specific framework. Use these to construct
test data or as base types for mock agents.
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FakeAgentConfig:
    """Minimal agent configuration for testing."""

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "orchestrator"
    name: str = "Test Agent"
    description: str = "A test agent"
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeAgentResponse:
    """Minimal agent response for testing.

    Satisfies AgentResponseProtocol.
    """

    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class FakeToolCall:
    """Represents a single tool invocation for testing."""

    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
