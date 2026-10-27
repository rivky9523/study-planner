"""Long-term memory: memory.json in the state dir (shared.store).

Written by the model through the save_memory tool (overwrite + keep history, so a
contradicting request replaces the old value). Read here, by the harness, to pick
only the memories relevant to the current message for the turn context.
"""
from API.harness import logger
from shared.memory_search import rank
from shared.store import Store

log = logger("memory.py")


class Memory:
    def __init__(self, store: Store):
        self.store = store

    def relevant(self, query: str, limit: int = 5) -> list[tuple[str, dict]]:
        entries = self.store.load_memory()["entries"]
        picked = rank(entries, query)[:limit]
        log(f"{len(entries)} entries on disk, {len(picked)} relevant: {[k for k, _ in picked]}")
        return picked

    @staticmethod
    def render(picked: list[tuple[str, dict]]) -> str:
        return "; ".join(f"{k}={e['value']}" for k, e in picked) or "(none)"
