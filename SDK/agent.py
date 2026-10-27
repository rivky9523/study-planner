"""The same study-planner agent on the Claude Agent SDK: Claude Code runs the
harness, this file only configures it. Lines tagged `# replaces: <part>` stand in
for API/harness/<part>.py. Compaction and caching have no lines - Claude Code does
both automatically."""
import datetime as dt
import time
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
                              ToolUseBlock)

from SDK.tools import allowed_tool_names, build_server
from shared.domain import SYSTEM_PROMPT, ToolContext
from shared.store import Store

MEMORY_NOTE = ("Use recall_memory for anything not listed here, and save_memory to add or change "
               "a preference (the newest statement wins).")


def write_claude_md(store: Store, state_dir: Path) -> None:
    """Pinned memory for this session: Claude Code loads ./CLAUDE.md (setting_sources=["project"])."""
    entries = store.load_memory()["entries"]  # replaces: memory
    lines = [f"- {k}: {e['value']} ({e['kind']})" for k, e in sorted(entries.items())]  # replaces: memory
    body = "\n".join(lines) or "- (nothing saved yet)"  # replaces: memory
    (state_dir / "CLAUDE.md").write_text(f"# Student memory\n{body}\n\n{MEMORY_NOTE}\n", encoding="utf-8")  # replaces: memory


class StudyAgentSDK:
    def __init__(self, state_dir, today: str, model: str = "claude-opus-5-5", effort: str = "medium"):
        self.state_dir = Path(state_dir)
        self.today, self.model, self.effort = today, model, effort
        self.store = Store(self.state_dir)
        self.server = build_server(ToolContext(self.store, today))
        self.session_id = None  # set by the first ResultMessage
        write_claude_md(self.store, self.state_dir)

    def options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=self.model,
            effort=self.effort,
            system_prompt=SYSTEM_PROMPT,  # replaces: context
            tools=[],  # no built-in tools (no Bash/Read/Write/Web...)
            mcp_servers={"study": self.server},  # replaces: tools
            allowed_tools=allowed_tool_names(),  # replaces: tools
            strict_mcp_config=True,
            permission_mode="dontAsk",
            resume=self.session_id,  # replaces: history
            setting_sources=["project"],  # replaces: memory
            cwd=str(self.state_dir),  # replaces: memory
            max_turns=12,  # replaces: loop
        )

    async def turn(self, user_text: str) -> dict:
        start = time.time()
        weekday = dt.date.fromisoformat(self.today).strftime("%A").lower()
        prompt = f"(today: {self.today} {weekday})\n{user_text}"  # replaces: context
        call_ids, tools_used, result = set(), [], None
        async with ClaudeSDKClient(options=self.options()) as client:  # replaces: loop
            await client.query(prompt)  # replaces: loop
            async for msg in client.receive_response():  # replaces: loop
                if isinstance(msg, AssistantMessage):
                    call_ids.add(msg.message_id)  # replaces: tokens
                    tools_used += [b.name.split("__")[-1] for b in msg.content if isinstance(b, ToolUseBlock)]
                elif isinstance(msg, ResultMessage):
                    result = msg
        self.session_id = result.session_id  # replaces: history
        u = result.usage or {}  # replaces: tokens
        return {
            "response": result.result or "",
            "tools": tools_used,
            "model_calls": len(call_ids),
            "tokens": {"input": u.get("input_tokens", 0),  # replaces: tokens
                       "cache_read": u.get("cache_read_input_tokens", 0),  # replaces: tokens
                       "cache_write": u.get("cache_creation_input_tokens", 0),  # replaces: tokens
                       "output": u.get("output_tokens", 0)},  # replaces: tokens
            "cost_usd": round(result.total_cost_usd or 0.0, 6),  # replaces: tokens
            "model_usage": result.model_usage,
            "seconds": round(time.time() - start, 2),
            "stop_reason": result.stop_reason,
            "sdk_turns": result.num_turns,
        }
