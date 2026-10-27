"""Domain tools for the study planner, shared by the API and SDK versions.

Every tool is a plain function `fn(ctx, **args) -> str`. Outputs are deliberately
compact (one line per item, short ids, pipe-separated) because every character a
tool returns is paid for as input tokens on every later model call in the turn.
Invalid input raises ToolError; each harness turns that into an is_error result.
"""
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from shared.memory_search import rank
from shared.store import Store

SYLLABI_DIR = Path(__file__).parent / "data" / "syllabi"
# Hebrew words -> syllabus words, so a Hebrew query still finds the English section.
QUERY_SYNONYMS = {
    "מבחן": "exam", "בחינה": "exam", "נושאים": "topics", "חומר": "topics",
    "ציון": "grading", "משקל": "grading", "אחוז": "grading", "אחוזים": "grading",
    "תרגיל": "homework exercise", "תרגילים": "homework exercises", "הגשה": "submission",
    "איחור": "late", "שיעורי": "homework", "מטלות": "homework exercises",
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
STUDY_TIMES = ["morning", "afternoon", "evening"]
DEFAULT_MAX_HOURS = 3.0


class ToolError(ValueError):
    """Bad tool input. The message goes back to the model so it can fix the call."""


@dataclass
class ToolContext:
    store: Store
    today: str  # YYYY-MM-DD, fixed by the scenario so runs are reproducible


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    fn: object


def _date(value: str, field: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ToolError(f"{field} must be YYYY-MM-DD, got {value!r}")


def _courses() -> list[str]:
    return sorted(p.stem for p in SYLLABI_DIR.glob("*.md"))


def _check_course(course: str) -> str:
    if course not in _courses():
        raise ToolError(f"unknown course {course!r}; known: {', '.join(_courses())}")
    return course


def _task_line(t: dict) -> str:
    return f"{t['id']}|{t['course']}|{t['title']}|due {t['due']}|{t['hours']:g}h|{t['status']}"


# ---------------------------------------------------------------- tasks

def add_task(ctx: ToolContext, course: str, title: str, due: str, hours: float) -> str:
    _check_course(course)
    _date(due, "due")
    if not 0 < float(hours) <= 40:
        raise ToolError("hours must be between 0 and 40")
    data = ctx.store.load_tasks()
    task = {"id": f"T{data['next_id']}", "course": course, "title": title.strip(),
            "due": due, "hours": float(hours), "status": "open"}
    data["next_id"] += 1
    data["tasks"].append(task)
    ctx.store.save_tasks(data)
    return "added " + _task_line(task)


def list_tasks(ctx: ToolContext, course: str | None = None, status: str = "open", limit: int = 10) -> str:
    if status not in ("open", "done", "all"):
        raise ToolError("status must be open, done or all")
    tasks = ctx.store.load_tasks()["tasks"]
    tasks = [t for t in tasks if (status == "all" or t["status"] == status)
             and (course is None or t["course"] == course)]
    tasks.sort(key=lambda t: t["due"])
    if not tasks:
        return "no tasks"
    lines = [_task_line(t) for t in tasks[:limit]]
    if len(tasks) > limit:
        lines.append(f"(+{len(tasks) - limit} more)")
    return "\n".join(lines)


def update_task(ctx: ToolContext, task_id: str, status: str | None = None,
                due: str | None = None, hours: float | None = None) -> str:
    data = ctx.store.load_tasks()
    task = next((t for t in data["tasks"] if t["id"] == task_id), None)
    if task is None:
        raise ToolError(f"no task {task_id}; call list_tasks to see ids")
    if status is not None:
        if status not in ("open", "done"):
            raise ToolError("status must be open or done")
        task["status"] = status
    if due is not None:
        _date(due, "due")
        task["due"] = due
    if hours is not None:
        task["hours"] = float(hours)
    ctx.store.save_tasks(data)
    return "updated " + _task_line(task)


# ---------------------------------------------------------------- external: syllabus files

def read_syllabus(ctx: ToolContext, course: str, query: str | None = None) -> str:
    """Reads a syllabus file from disk. With a query, returns only the matching
    section(s) instead of the whole file - context engineering at the tool level."""
    _check_course(course)
    text = (SYLLABI_DIR / f"{course}.md").read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)
    if not query:
        heads = [s.splitlines()[0].lstrip("# ") for s in sections]
        return sections[0].strip() + "\nsections: " + "; ".join(heads[1:])
    words = []
    for w in re.findall(r"\w+", query.lower()):
        bare = w[1:] if w[:1] in "ובהלמש" else w  # drop one Hebrew prefix letter (ה, ב, ו...)
        words += QUERY_SYNONYMS.get(w, QUERY_SYNONYMS.get(bare, w)).split()
    words = [w for w in words if len(w) > 2]
    hits = [s.strip() for s in sections[1:] if any(w in s.lower() for w in words)]
    if not hits:
        # Syllabi are short: better to return the whole file than nothing.
        return f"no section matched {query!r}; full syllabus:\n{text.strip()}"
    return "\n".join(hits)


# ---------------------------------------------------------------- memory

def _validate_memory(key: str, value: str) -> str:
    v = value.strip().lower()
    if key == "study_time" and v not in STUDY_TIMES:
        raise ToolError(f"study_time must be one of {STUDY_TIMES}")
    if key == "max_hours_per_day":
        try:
            float(v)
        except ValueError:
            raise ToolError("max_hours_per_day must be a number")
    if key == "days_off":
        days = [d.strip() for d in v.split(",") if d.strip()]
        bad = [d for d in days if d not in WEEKDAYS]
        if bad:
            raise ToolError(f"days_off must be english weekday names, bad: {bad}")
        v = ",".join(days)
    return v if key in ("study_time", "max_hours_per_day", "days_off") else value.strip()


def save_memory(ctx: ToolContext, key: str, value: str, kind: str = "preference") -> str:
    if kind not in ("preference", "decision", "fact"):
        raise ToolError("kind must be preference, decision or fact")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", key):
        raise ToolError("key must be snake_case")
    value = _validate_memory(key, value)
    data = ctx.store.load_memory()
    old = data["entries"].get(key)
    history = old["history"] if old else []
    if old and old["value"] != value:
        # A contradicting request replaces the value but keeps the old one as history.
        history = history + [{"value": old["value"], "replaced_on": ctx.today}]
    data["entries"][key] = {"value": value, "kind": kind, "updated": ctx.today, "history": history}
    ctx.store.save_memory(data)
    note = f" (replaced: {old['value']})" if old and old["value"] != value else ""
    return f"saved {key}={value}{note}"


def recall_memory(ctx: ToolContext, query: str, limit: int = 5) -> str:
    entries = ctx.store.load_memory()["entries"]
    ranked = rank(entries, query)[:limit]
    if not ranked:
        return "nothing relevant in memory"
    return "\n".join(f"{k}={e['value']} ({e['kind']}, {e['updated']})" for k, e in ranked)


# ---------------------------------------------------------------- planning

def preferences(store: Store) -> dict:
    entries = store.load_memory()["entries"]
    get = lambda k, d: entries[k]["value"] if k in entries else d
    return {
        "study_time": get("study_time", "evening"),
        "max_hours_per_day": float(get("max_hours_per_day", DEFAULT_MAX_HOURS)),
        "days_off": [d for d in get("days_off", "").split(",") if d],
    }


def plan_week(ctx: ToolContext, start: str | None = None, days: int = 7) -> str:
    """Greedy earliest-due-first schedule of open tasks, honouring the saved
    preferences (study_time, max_hours_per_day, days_off). Saves plan.json."""
    start_d = _date(start or ctx.today, "start")
    prefs = preferences(ctx.store)
    tasks = sorted((t for t in ctx.store.load_tasks()["tasks"] if t["status"] == "open"),
                   key=lambda t: t["due"])
    remaining = {t["id"]: t["hours"] for t in tasks}
    plan_days = []
    for i in range(days):
        d = start_d + dt.timedelta(days=i)
        weekday = WEEKDAYS[d.weekday()]
        items, free = [], prefs["max_hours_per_day"]
        if weekday not in prefs["days_off"]:
            for t in tasks:
                if free <= 0:
                    break
                if remaining[t["id"]] > 0 and d < dt.date.fromisoformat(t["due"]):
                    h = min(free, remaining[t["id"]])
                    remaining[t["id"]] -= h
                    free -= h
                    items.append({"task": t["id"], "hours": h})
        plan_days.append({"date": d.isoformat(), "weekday": weekday, "items": items})
    unscheduled = {k: v for k, v in remaining.items() if v > 0}
    ctx.store.save_plan({"start": start_d.isoformat(), "slot": prefs["study_time"],
                         "max_hours_per_day": prefs["max_hours_per_day"],
                         "days": plan_days, "unscheduled": unscheduled})
    lines = [f"slot={prefs['study_time']} max={prefs['max_hours_per_day']:g}h/day "
             f"off={','.join(prefs['days_off']) or '-'}"]
    for day in plan_days:
        if day["items"]:
            work = ", ".join(f"{it['task']} {it['hours']:g}h" for it in day["items"])
        else:
            work = "off" if day["weekday"] in prefs["days_off"] else "free"
        lines.append(f"{day['date']} {day['weekday'][:3]}: {work}")
    if unscheduled:
        lines.append("NOT FITTING before due: " + ", ".join(f"{k} {v:g}h" for k, v in unscheduled.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- registry

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


COURSE = {"type": "string", "description": "course id, e.g. linear-algebra, intro-cs"}

TOOLS: list[ToolSpec] = [
    ToolSpec("add_task", "Add a homework/study task. Returns the new task line.",
             _obj({"course": COURSE, "title": {"type": "string"},
                   "due": {"type": "string", "description": "YYYY-MM-DD"},
                   "hours": {"type": "number", "description": "estimated hours"}},
                  ["course", "title", "due", "hours"]), add_task),
    ToolSpec("list_tasks", "List tasks sorted by due date, one compact line each: id|course|title|due|hours|status.",
             _obj({"course": COURSE, "status": {"type": "string", "enum": ["open", "done", "all"]},
                   "limit": {"type": "integer"}}, []), list_tasks),
    ToolSpec("update_task", "Mark a task done/open, or change its due date or hours.",
             _obj({"task_id": {"type": "string"}, "status": {"type": "string", "enum": ["open", "done"]},
                   "due": {"type": "string"}, "hours": {"type": "number"}}, ["task_id"]), update_task),
    ToolSpec("read_syllabus",
             "Read a course syllabus file (written in English). Pass an English query (e.g. 'exam topics', "
             "'grading', 'homework') to get only the matching sections; without a query you get the "
             "header and the list of sections.",
             _obj({"course": COURSE, "query": {"type": "string"}}, ["course"]), read_syllabus),
    ToolSpec("save_memory",
             "Persist a long-term preference/decision/fact across sessions. Saving an existing key replaces "
             "its value (the old value is kept as history). Canonical preference keys: study_time "
             "(morning|afternoon|evening), max_hours_per_day (number), days_off (comma-separated english "
             "weekdays, e.g. saturday). Use other snake_case keys for anything else.",
             _obj({"key": {"type": "string"}, "value": {"type": "string"},
                   "kind": {"type": "string", "enum": ["preference", "decision", "fact"]}},
                  ["key", "value"]), save_memory),
    ToolSpec("recall_memory", "Search long-term memory by keywords. Returns key=value lines.",
             _obj({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]), recall_memory),
    ToolSpec("plan_week",
             "Build and save a study schedule for open tasks from start date (default today), honouring the "
             "saved preferences. Returns one line per day.",
             _obj({"start": {"type": "string", "description": "YYYY-MM-DD"},
                   "days": {"type": "integer"}}, []), plan_week),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a personal study-planning assistant for a university student.
You manage their tasks, read course syllabi, remember their preferences across sessions, and build weekly study plans.

How to work:
- Use the tools for every fact about tasks, syllabi or memory; never invent due dates or syllabus content.
- When the student states a lasting preference or decision, save it with save_memory right away. If it contradicts something saved earlier, overwrite the same key - the new statement wins.
- When preferences change, rebuild the plan with plan_week so it reflects them.
- Ask a short question only if a required detail (like a due date) is missing.
- Reply in the student's language, briefly: a few lines, no long preambles."""
