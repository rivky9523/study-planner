"""Persistent domain state (tasks, memory, weekly plan) as JSON files in a state dir.

Both the API and SDK versions use the same store, so the scenario checks can
inspect identical files regardless of which harness ran.
"""
import json
import os
import tempfile
from pathlib import Path


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    # Atomic write: a crash mid-write never leaves a half-written state file.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


class Store:
    def __init__(self, state_dir):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.dir / "tasks.json"
        self.memory_path = self.dir / "memory.json"
        self.plan_path = self.dir / "plan.json"

    # tasks: {"next_id": int, "tasks": [{id, course, title, due, hours, status}]}
    def load_tasks(self) -> dict:
        return _read(self.tasks_path, {"next_id": 1, "tasks": []})

    def save_tasks(self, data: dict) -> None:
        _write(self.tasks_path, data)

    # memory: {"entries": {key: {value, kind, updated, history: [{value, replaced_on}]}}}
    def load_memory(self) -> dict:
        return _read(self.memory_path, {"entries": {}})

    def save_memory(self, data: dict) -> None:
        _write(self.memory_path, data)

    def load_plan(self) -> dict | None:
        return _read(self.plan_path, None)

    def save_plan(self, data: dict) -> None:
        _write(self.plan_path, data)
