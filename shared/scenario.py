"""Scenario loading, pass/fail checks and results writing - identical for both versions."""
import ast
import json
import re
import shutil
from pathlib import Path

from shared.store import Store

ROOT = Path(__file__).resolve().parent.parent
HARNESS_PARTS = ["loop", "tools", "history", "memory", "context", "compaction", "caching", "tokens"]


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_state(state_dir) -> Path:
    """Every run starts from a clean state."""
    p = Path(state_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


# ---------------------------------------------------------------- checks

def _find_task(tasks, course, title_contains):
    for t in tasks:
        if t["course"] == course and title_contains.lower() in t["title"].lower():
            return t
    return None


def _turn(logs, turn_id):
    return next((t for t in logs if t["turn"] == turn_id), None)


def run_check(check: dict, store: Store, logs: list[dict]) -> tuple[bool, str]:
    kind = check["type"]
    tasks = store.load_tasks()["tasks"]
    memory = store.load_memory()["entries"]
    if kind == "task_exists":
        t = _find_task(tasks, check["course"], check.get("title_contains", ""))
        ok = t is not None and ("due" not in check or t["due"] == check["due"])
        return ok, json.dumps(t, ensure_ascii=False) if t else "task not found"
    if kind == "task_status":
        t = _find_task(tasks, check["course"], check.get("title_contains", ""))
        return (t is not None and t["status"] == check["status"]), (t["status"] if t else "task not found")
    if kind == "memory_equals":
        e = memory.get(check["key"])
        return (e is not None and str(e["value"]).lower() == str(check["value"]).lower()), \
            (str(e["value"]) if e else "missing")
    if kind == "memory_contains":
        e = memory.get(check["key"])
        return (e is not None and check["value"].lower() in str(e["value"]).lower()), \
            (str(e["value"]) if e else "missing")
    if kind == "memory_history_contains":
        e = memory.get(check["key"])
        old = [h["value"] for h in e["history"]] if e else []
        return check["value"] in old, f"history={old}"
    if kind == "plan":
        plan = store.load_plan()
        if not plan:
            return False, "no plan saved"
        problems = []
        if "slot" in check and plan["slot"] != check["slot"]:
            problems.append(f"slot={plan['slot']}")
        for d in plan["days"]:
            if d["weekday"] == check.get("no_weekday") and d["items"]:
                problems.append(f"work on {d['date']}")
            if sum(i["hours"] for i in d["items"]) > check.get("max_hours_per_day", 99):
                problems.append(f"too much on {d['date']}")
        return not problems, "; ".join(problems) or f"slot={plan['slot']}"
    if kind == "tool_called":
        t = _turn(logs, check["turn"])
        tools = t["tools"] if t else []
        return check["tool"] in tools, f"tools={tools}"
    if kind == "response_contains_any":
        t = _turn(logs, check["turn"])
        text = t["response"] if t else ""
        return any(w in text for w in check["any"]), text[:120].replace("\n", " ")
    raise ValueError(f"unknown check type {kind}")


def run_checks(scenario: dict, store: Store, logs: list[dict]) -> list[dict]:
    out = []
    for c in scenario["checks"]:
        ok, detail = run_check(c, store, logs)
        out.append({"id": c["id"], "description": c["description"], "pass": ok, "detail": detail})
    return out


# ---------------------------------------------------------------- lines of code

def _code_lines(path: Path) -> list[str]:
    """Source lines that are code: no blank lines, comment-only lines or docstrings."""
    src = path.read_text(encoding="utf-8")
    doc_lines = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return [line for i, line in enumerate(src.splitlines(), 1)
            if i not in doc_lines and line.strip() and not line.strip().startswith("#")]


def loc_api() -> dict:
    d = ROOT / "API" / "harness"
    return {part: len(_code_lines(d / f"{part}.py")) for part in HARNESS_PARTS}


def loc_sdk() -> dict:
    """Counts the SDK lines tagged `# replaces: <part>` (a line may replace several parts)."""
    counts = {part: 0 for part in HARNESS_PARTS}
    for f in (ROOT / "SDK").glob("*.py"):
        for line in _code_lines(f):
            m = re.search(r"#\s*replaces:\s*([\w, ]+)", line)
            if m:
                for part in re.split(r"[,\s]+", m.group(1).strip()):
                    if part in counts:
                        counts[part] += 1
    return counts


# ---------------------------------------------------------------- results

def summarize(logs: list[dict]) -> dict:
    tot = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    for t in logs:
        for k in tot:
            tot[k] += t["tokens"][k]
    all_in = tot["input"] + tot["cache_read"] + tot["cache_write"]
    return {
        "tokens": tot,
        "cache_hit_pct": round(100 * tot["cache_read"] / all_in, 1) if all_in else 0.0,
        "model_calls": sum(t["model_calls"] for t in logs),
        "cost_usd": round(sum(t["cost_usd"] for t in logs), 4),
        "seconds": round(sum(t["seconds"] for t in logs), 1),
    }


def write_results(out_path, version: str, scenario: dict, logs: list[dict], checks: list[dict],
                  loc: dict, extra: dict | None = None) -> dict:
    result = {
        "version": version,
        "model": scenario["model"],
        "effort": scenario["effort"],
        "turns": logs,
        "summary": summarize(logs),
        "checks": checks,
        "checks_passed": f"{sum(c['pass'] for c in checks)}/{len(checks)}",
        "harness_loc": loc,
        **(extra or {}),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
