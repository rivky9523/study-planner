"""Client-side "simple" compaction: once the context passes a threshold, summarize
the whole conversation into one message and start the next request from that
summary + the new user turn. No earlier turns (or their thinking blocks) are
replayed, so nothing carried over is tied to the old transcript. Runs only between
turns, never in the middle of a tool round."""
from API.harness import logger

log = logger("compaction.py")

SUMMARY_PROMPT = (
    "Summarize this conversation so far so it can replace the full history. Keep: tasks discussed "
    "(ids, courses, due dates), every preference or decision the student stated (and which ones were "
    "changed), plans that were built, open questions and anything promised for later. Drop small talk. "
    "Answer with the summary text only - do not call any tools."
)


def should_compact(context_tokens: int, threshold: int) -> bool:
    if context_tokens > threshold:
        log(f"context {context_tokens} tokens > threshold {threshold}: compacting")
        return True
    return False


def summarize(client, build_request, messages: list[dict]):
    # Same system + tools as the conversation, so this call still reads the cached prefix.
    req = build_request(messages + [{"role": "user", "content": [{"type": "text", "text": SUMMARY_PROMPT}]}])
    resp = client.beta.messages.create(**req)
    summary = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    log(f"summarized {len(messages)} messages into {len(summary)} chars")
    return summary, resp


def summary_block(summary: str) -> dict:
    return {"type": "text", "text": f"<conversation_summary>\n{summary}\n</conversation_summary>"}
