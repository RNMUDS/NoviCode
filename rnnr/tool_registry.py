"""Tool registry — instantiates and dispatches tool calls."""

from __future__ import annotations

from typing import Any

from rnnr.config import ModeProfile
from rnnr.security_manager import SecurityManager
from rnnr.policy_engine import PolicyEngine
from rnnr.tools.bash_tool import BashTool
from rnnr.tools.read_tool import ReadTool
from rnnr.tools.write_tool import WriteTool
from rnnr.tools.edit_tool import EditTool
from rnnr.tools.grep_tool import GrepTool
from rnnr.tools.glob_tool import GlobTool


class ToolRegistry:
    """Manages available tools and dispatches calls with policy checks."""

    def __init__(
        self,
        security: SecurityManager,
        policy: PolicyEngine,
        profile: ModeProfile,
        working_dir: str,
    ) -> None:
        self.security = security
        self.policy = policy
        self.profile = profile
        self._tools: dict[str, Any] = {}
        self._register(working_dir)

    def _register(self, working_dir: str) -> None:
        all_tools = {
            "bash": BashTool(self.security, working_dir),
            "read": ReadTool(self.security, working_dir),
            "write": WriteTool(self.security, self.policy, working_dir),
            "edit": EditTool(self.security, self.policy, working_dir),
            "grep": GrepTool(self.security, working_dir),
            "glob": GlobTool(self.security, working_dir),
        }
        # Only register tools allowed by the mode profile
        for name, tool in all_tools.items():
            if name in self.profile.allowed_tools:
                self._tools[name] = tool

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call, returning a result dict."""
        # Policy check
        verdict = self.policy.check_tool_allowed(tool_name)
        if not verdict.allowed:
            return {"error": verdict.reason}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        return tool.execute(arguments)

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())
