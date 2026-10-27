"""What goes into each turn. The system prompt and tools never change (cached);
the volatile, per-turn facts go in a small <context> block inside the user message:
today's date, only the relevant memories, and a short open-task digest (the model
calls list_tasks / read_syllabus when it needs more)."""
import datetime as dt

from API.harness import logger
from API.harness.memory import Memory
from shared.store import Store

log = logger("context.py")
DIGEST_TASKS = 3


def build(store: Store, memory: Memory, today: str, user_text: str) -> str:
    weekday = dt.date.fromisoformat(today).strftime("%A").lower()
    open_tasks = sorted((t for t in store.load_tasks()["tasks"] if t["status"] == "open"),
                        key=lambda t: t["due"])
    digest = "; ".join(f"{t['id']} {t['course']} \"{t['title']}\" due {t['due']}" for t in open_tasks[:DIGEST_TASKS])
    more = f" (+{len(open_tasks) - DIGEST_TASKS} more, use list_tasks)" if len(open_tasks) > DIGEST_TASKS else ""
    block = (f"<context>\ntoday: {today} {weekday}\n"
             f"memory: {Memory.render(memory.relevant(user_text))}\n"
             f"open_tasks: {len(open_tasks)}{(' | next: ' + digest) if digest else ''}{more}\n</context>")
    log(f"context block {len(block)} chars, {min(len(open_tasks), DIGEST_TASKS)}/{len(open_tasks)} tasks inlined")
    return block
