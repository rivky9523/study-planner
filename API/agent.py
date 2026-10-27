"""The study-planner agent on the raw Messages API - wires the harness files together."""
import time

from API.harness import caching, compaction, context, loop, tokens, tools
from API.harness.history import History
from API.harness.memory import Memory
from shared.domain import SYSTEM_PROMPT, ToolContext
from shared.store import Store

# Server-side refusal fallback (beta): if the model declines, the API reroutes by itself.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class StudyAgent:
    def __init__(self, client, state_dir, session_id: str, today: str,
                 model: str = "claude-opus-5-5", effort: str = "medium", compact_at: int = 20_000):
        self.client, self.model, self.effort, self.compact_at = client, model, effort, compact_at
        self.today = today
        self.store = Store(state_dir)
        self.tool_ctx = ToolContext(self.store, today)
        self.history = History(state_dir, session_id)
        self.memory = Memory(self.store)

    def _request(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16_000,
            "system": caching.system_blocks(SYSTEM_PROMPT),
            "tools": tools.schemas(),
            "messages": caching.mark_last(messages),
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def turn(self, user_text: str) -> dict:
        start, usage = time.time(), tokens.TurnUsage()
        messages = self.history.messages()
        summary = self.history.pending_summary()

        if messages and compaction.should_compact(self.history.last_context_tokens(), self.compact_at):
            summary, resp = compaction.summarize(self.client, self._request, messages)
            usage.add(resp.usage)
            self.history.append({"type": "compaction", "summary": summary})
            messages = []

        content = [compaction.summary_block(summary)] if summary else []
        content += [{"type": "text", "text": context.build(self.store, self.memory, self.today, user_text)},
                    {"type": "text", "text": user_text}]
        user_msg = {"role": "user", "content": content}
        self.history.append_message(user_msg)

        result = loop.run(self.client, self._request, messages + [user_msg],
                          lambda uses: tools.execute(self.tool_ctx, uses),
                          self.history.append_message)
        for resp in result.responses:
            usage.add(resp.usage)
        self.history.append({"type": "usage", "context_tokens": tokens.context_tokens(result.responses[-1].usage)})

        return {
            "response": result.text,
            "tools": result.tools_used,
            "model_calls": usage.calls,
            "tokens": usage.tokens(),
            "cost_usd": usage.cost(self.model),
            "seconds": round(time.time() - start, 2),
            "stop_reason": result.stop_reason,
        }
