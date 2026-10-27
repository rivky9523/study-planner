"""Conversation history: an append-only JSONL file per session on disk.

Records: {"type": "message", "role", "content"} | {"type": "compaction", "summary"}
| {"type": "usage", "context_tokens"}. Nothing is ever rewritten - compaction is a
new record, and the model sees only the messages after the latest one. Keeping
the sent history byte-identical also keeps the prompt cache and thinking blocks valid.
"""
import json
from pathlib import Path

from API.harness import logger

log = logger("history.py")


class History:
    def __init__(self, state_dir, session_id: str):
        self.path = Path(state_dir) / "sessions" / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log(f"session {session_id}: {len(self.records())} records on disk")

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"append {record['type']}" + (f" ({record['role']})" if "role" in record else ""))

    def append_message(self, msg: dict) -> None:
        self.append({"type": "message", "role": msg["role"], "content": msg["content"]})

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _since_compaction(self) -> tuple[list[dict], str | None]:
        msgs, summary = [], None
        for r in self.records():
            if r["type"] == "compaction":
                msgs, summary = [], r["summary"]
            elif r["type"] == "message":
                msgs.append({"role": r["role"], "content": r["content"]})
        return msgs, summary

    def messages(self) -> list[dict]:
        return self._since_compaction()[0]

    def pending_summary(self) -> str | None:
        """A summary not yet sent to the model (compaction was the last thing recorded)."""
        msgs, summary = self._since_compaction()
        return summary if not msgs else None

    def last_context_tokens(self) -> int:
        usage = [r for r in self.records() if r["type"] == "usage"]
        return usage[-1]["context_tokens"] if usage else 0
