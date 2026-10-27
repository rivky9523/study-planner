"""The agent loop: call the model while stop_reason == "tool_use"."""
from dataclasses import dataclass, field

from API.harness import logger

log = logger("loop.py")
MAX_ITERATIONS = 12


@dataclass
class LoopResult:
    text: str = ""
    responses: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    stop_reason: str = ""


def _as_dict(block) -> dict:
    # Assistant content goes back exactly as received (thinking blocks included).
    return block.to_dict() if hasattr(block, "to_dict") else dict(block)


def run(client, build_request, messages: list[dict], execute_tools, on_message) -> LoopResult:
    """messages ends with the new user message. on_message(msg) is called for every
    message the loop appends, so history is written as the turn progresses."""
    messages = list(messages)
    result = LoopResult()
    for i in range(1, MAX_ITERATIONS + 1):
        log(f"model call #{i} ({len(messages)} messages)")
        resp = client.beta.messages.create(**build_request(messages))
        result.responses.append(resp)
        result.stop_reason = resp.stop_reason
        assistant = {"role": "assistant", "content": [_as_dict(b) for b in resp.content]}
        messages.append(assistant)
        on_message(assistant)
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        log(f"stop_reason={resp.stop_reason} tool_calls={[b.name for b in tool_uses]}")

        if resp.stop_reason == "tool_use":
            result.tools_used += [b.name for b in tool_uses]
            tool_results = execute_tools(tool_uses)
            # All tool_results of this step go back in ONE user message.
            msg = {"role": "user", "content": tool_results}
            messages.append(msg)
            on_message(msg)
            continue

        if tool_uses:
            # e.g. max_tokens cut the response after a tool_use: every tool_use
            # must still get a result or the next request is rejected.
            msg = {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                 "content": f"not executed: response stopped ({resp.stop_reason})"} for b in tool_uses]}
            messages.append(msg)
            on_message(msg)
        break
    else:
        log(f"stopped after MAX_ITERATIONS={MAX_ITERATIONS}")

    result.text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "refusal" and not result.text:
        result.text = "[the model declined this request]"
    return result
