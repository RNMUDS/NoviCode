"""Tool registry — instantiates and dispatches tool calls."""

from __future__ import annotations

from typing import Any

from novicode.config import ModeProfile
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine
from novicode.tools.bash_tool import BashTool
from novicode.tools.read_tool import ReadTool
from novicode.tools.write_tool import WriteTool
from novicode.tools.edit_tool import EditTool
from novicode.tools.grep_tool import GrepTool
from novicode.tools.glob_tool import GlobTool


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
            "bash": BashTool(self.security, working_dir, mode=self.profile.mode),
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
