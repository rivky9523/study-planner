import pytest

from shared import scenario as sc
from shared.domain import (ToolContext, ToolError, add_task, list_tasks, plan_week, read_syllabus,
                           recall_memory, save_memory, update_task)
from shared.store import Store


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(Store(tmp_path), "2026-11-01")


def test_add_and_list_are_compact(ctx):
    out = add_task(ctx, "linear-algebra", "HW3", "2026-11-05", 3)
    assert out == "added T1|linear-algebra|HW3|due 2026-11-05|3h|open"
    for i in range(4):
        add_task(ctx, "intro-cs", f"ex{i}", f"2026-11-1{i}", 1)
    listing = list_tasks(ctx, limit=3)
    assert listing.splitlines()[-1] == "(+2 more)"
    assert len(listing.splitlines()) == 4


def test_unknown_course_lists_known_ones(ctx):
    with pytest.raises(ToolError, match="known: intro-cs, linear-algebra"):
        add_task(ctx, "calculus", "x", "2026-11-05", 1)


def test_bad_date_rejected(ctx):
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        add_task(ctx, "intro-cs", "x", "5.11", 1)


def test_update_task(ctx):
    add_task(ctx, "intro-cs", "ex2", "2026-11-04", 2)
    assert update_task(ctx, "T1", status="done").endswith("|done")
    with pytest.raises(ToolError):
        update_task(ctx, "T9", status="done")


def test_syllabus_query_returns_only_matching_section(ctx):
    out = read_syllabus(ctx, "linear-algebra", "exam topics")
    assert "Gaussian elimination" in out
    assert "late submission" not in out
    head = read_syllabus(ctx, "linear-algebra")
    assert "sections: Grading; Exam topics; Homework policy" in head


def test_contradiction_overwrites_and_keeps_history(ctx):
    save_memory(ctx, "study_time", "morning")
    out = save_memory(ctx, "study_time", "Evening")
    assert out == "saved study_time=evening (replaced: morning)"
    entry = ctx.store.load_memory()["entries"]["study_time"]
    assert entry["value"] == "evening"
    assert entry["history"] == [{"value": "morning", "replaced_on": "2026-11-01"}]


def test_memory_validation(ctx):
    with pytest.raises(ToolError):
        save_memory(ctx, "study_time", "בבוקר")
    with pytest.raises(ToolError):
        save_memory(ctx, "days_off", "shabbat")
    assert save_memory(ctx, "days_off", "Saturday") == "saved days_off=saturday"


def test_recall_finds_preferences_from_hebrew_query(ctx):
    save_memory(ctx, "study_time", "morning")
    save_memory(ctx, "exam_strategy", "solve old exams first", "decision")
    out = recall_memory(ctx, "מתי נוח לי ללמוד בבוקר?")
    assert out.splitlines()[0].startswith("study_time=morning")
    assert "exam_strategy" not in out


def test_plan_respects_preferences(ctx):
    add_task(ctx, "linear-algebra", "HW3", "2026-11-05", 3)
    add_task(ctx, "intro-cs", "ex2", "2026-11-04", 2)
    save_memory(ctx, "max_hours_per_day", "2")
    save_memory(ctx, "days_off", "monday")
    save_memory(ctx, "study_time", "evening")
    out = plan_week(ctx)
    plan = ctx.store.load_plan()
    assert plan["slot"] == "evening"
    by_day = {d["weekday"]: d for d in plan["days"]}
    assert by_day["monday"]["items"] == []
    assert all(sum(i["hours"] for i in d["items"]) <= 2 for d in plan["days"])
    # earliest due first: ex2 (T2) gets Sunday
    assert by_day["sunday"]["items"] == [{"task": "T2", "hours": 2.0}]
    assert "mon: off" in out
    assert plan["unscheduled"] == {}


def test_checks_pass_and_fail_on_state(ctx):
    scen = sc.load(sc.ROOT / "scenario.json")
    add_task(ctx, "linear-algebra", "תרגיל 3", "2026-11-05", 3)
    add_task(ctx, "intro-cs", "תרגיל 2", "2026-11-04", 2)
    update_task(ctx, "T2", status="done")
    save_memory(ctx, "study_time", "morning")
    save_memory(ctx, "study_time", "evening")
    save_memory(ctx, "max_hours_per_day", "3")
    save_memory(ctx, "days_off", "saturday")
    plan_week(ctx)
    logs = [{"turn": "t2", "tools": ["read_syllabus"], "response": ""},
            {"turn": "t6", "tools": [], "response": "אמרת שנוח לך בבוקר, עד 3 שעות"}]
    results = {c["id"]: c["pass"] for c in sc.run_checks(scen, ctx.store, logs)}
    assert all(results.values()), results
    save_memory(ctx, "study_time", "morning")
    results = {c["id"]: c["pass"] for c in sc.run_checks(scen, ctx.store, logs[:1])}
    assert not results["c3"] and not results["c9"]


def test_loc_counts():
    api, sdk = sc.loc_api(), sc.loc_sdk()
    assert set(api) == set(sdk) == set(sc.HARNESS_PARTS)
    assert all(v > 0 for v in api.values())
    assert sdk["compaction"] == 0 and sdk["caching"] == 0
    assert sdk["tools"] > 0 and sdk["history"] > 0 and sdk["memory"] > 0


def test_syllabus_hebrew_query(ctx):
    out = read_syllabus(ctx, "linear-algebra", "אילו נושאים יהיו במבחן")
    assert "Gaussian elimination" in out
    grading = read_syllabus(ctx, "linear-algebra", "המשקל של התרגילים")
    assert "homework: 10%" in grading


def test_syllabus_no_match_returns_whole_file(ctx):
    out = read_syllabus(ctx, "intro-cs", "zzz")
    assert out.startswith("no section matched") and "binary search" in out
