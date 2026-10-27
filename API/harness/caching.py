"""Prompt-cache breakpoints. Render order is tools -> system -> messages:
 1. end of the system prompt  -> caches tools + system (identical on every call)
 2. last block of the request -> caches the whole conversation so far, so the
    next call (next tool step or next turn) reads it from cache.
Markers are added to a copy at request time and never stored in history."""
from API.harness import logger

log = logger("caching.py")
EPHEMERAL = {"type": "ephemeral"}


def system_blocks(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": EPHEMERAL}]


def mark_last(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    content = list(content)
    content[-1] = {**content[-1], "cache_control": EPHEMERAL}
    last["content"] = content
    log(f"breakpoints: system + message #{len(messages)} ({content[-1]['type']})")
    return messages[:-1] + [last]
