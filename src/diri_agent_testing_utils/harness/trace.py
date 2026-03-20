"""
Invocation trace for recording agent execution details.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class InvocationTrace:
    """Records the details of a single agent invocation.

    Captures input, output, tool calls, timing, and metadata
    for test assertions.
    """

    input_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    response_content: str = ""
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    duration_ms: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error: str = ""
