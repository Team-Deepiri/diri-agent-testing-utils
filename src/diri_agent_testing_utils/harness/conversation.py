"""
Conversation scripting for agent test harness.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationStep:
    """A single step in a scripted agent conversation.

    Defines the input to send and optional assertions on the response.
    """

    input_text: str
    context: Optional[Dict[str, Any]] = None
    expected_contains: Optional[List[str]] = None
    expected_tool_calls: Optional[List[str]] = None
    min_confidence: Optional[float] = None


@dataclass
class ConversationScript:
    """A sequence of conversation steps for testing an agent.

    Example::

        script = ConversationScript(
            name="basic_greeting",
            steps=[
                ConversationStep(
                    input_text="Hello!",
                    expected_contains=["hello", "hi"],
                ),
                ConversationStep(
                    input_text="What tools do you have?",
                    expected_tool_calls=["list_tools"],
                ),
            ],
        )
    """

    steps: List[ConversationStep] = field(default_factory=list)
    name: str = ""
