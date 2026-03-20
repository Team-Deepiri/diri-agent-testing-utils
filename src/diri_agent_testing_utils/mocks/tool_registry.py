"""
Fake tool registry for testing agent tool execution.
"""
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class FakeToolRegistry:
    """In-memory tool registry that tracks all tool executions.

    Provides register_tool, get_tool, list_tools, and execute methods
    with full call logging for test assertions.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[Dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
    ) -> None:
        """Register a tool function."""
        self._tools[name] = {
            "func": func,
            "description": description or (func.__doc__ or "No description"),
        }

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """Get a registered tool function by name."""
        tool = self._tools.get(name)
        return tool["func"] if tool else None

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get all tool names and descriptions."""
        return {name: info["description"] for name, info in self._tools.items()}

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool and log the call."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not registered")

        result = tool["func"](**kwargs)
        self.call_log.append({
            "tool_name": name,
            "kwargs": kwargs,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    async def aexecute(self, name: str, **kwargs: Any) -> Any:
        """Async execute a registered tool and log the call."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not registered")

        result = tool["func"](**kwargs)
        # Support awaitable results
        if hasattr(result, "__await__"):
            result = await result

        self.call_log.append({
            "tool_name": name,
            "kwargs": kwargs,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return result

    def get_calls_for(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get all logged calls for a specific tool."""
        return [c for c in self.call_log if c["tool_name"] == tool_name]

    def reset(self) -> None:
        """Clear all tools and call log."""
        self._tools.clear()
        self.call_log.clear()
