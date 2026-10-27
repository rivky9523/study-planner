"""The shared domain tools exposed to Claude Code as an in-process MCP server."""
from claude_agent_sdk import create_sdk_mcp_server, tool

from shared.domain import TOOLS, ToolContext, ToolError, ToolSpec

SERVER = "study"


def _wrap(ctx: ToolContext, spec: ToolSpec):
    @tool(spec.name, spec.description, spec.input_schema)  # replaces: tools
    async def handler(args: dict) -> dict:
        try:
            return {"content": [{"type": "text", "text": spec.fn(ctx, **args)}]}  # replaces: tools
        except (ToolError, TypeError) as e:
            return {"content": [{"type": "text", "text": str(e)}], "is_error": True}  # replaces: tools
    return handler


def build_server(ctx: ToolContext):
    return create_sdk_mcp_server(SERVER, tools=[_wrap(ctx, s) for s in TOOLS])  # replaces: tools


def allowed_tool_names() -> list[str]:
    return [f"mcp__{SERVER}__{s.name}" for s in TOOLS]  # replaces: tools
