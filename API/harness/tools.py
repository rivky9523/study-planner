"""Tool schemas (JSON) + dispatch. Errors go back to the model as is_error results."""
import json

from API.harness import logger
from shared.domain import TOOLS, TOOLS_BY_NAME, ToolContext, ToolError

log = logger("tools.py")


def schemas() -> list[dict]:
    # Fixed order, built once per request from the same registry -> byte-identical
    # tools prefix on every call, which the prompt cache depends on.
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in TOOLS]


def dispatch(ctx: ToolContext, name: str, args: dict) -> tuple[str, bool]:
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return f"unknown tool {name}", True
    try:
        out = spec.fn(ctx, **args)
        log(f"{name}({json.dumps(args, ensure_ascii=False)}) -> {out.splitlines()[0][:80]}")
        return out, False
    except ToolError as e:
        log(f"{name} rejected input: {e}")
        return str(e), True
    except TypeError as e:  # wrong/missing arguments
        log(f"{name} bad arguments: {e}")
        return f"bad arguments: {e}", True
    except Exception as e:  # a tool bug must not kill the agent loop
        log(f"{name} crashed: {type(e).__name__}: {e}")
        return f"internal error: {type(e).__name__}: {e}", True


def execute(ctx: ToolContext, tool_uses) -> list[dict]:
    """Runs every tool_use of one assistant message; all results return together."""
    results = []
    for block in tool_uses:
        content, is_error = dispatch(ctx, block.name, block.input or {})
        result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
        if is_error:
            result["is_error"] = True
        results.append(result)
    return results
