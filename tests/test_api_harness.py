import json

from API.agent import StudyAgent
from API.harness import caching, loop, tokens
from API.harness.history import History
from API.main import run as run_scenario
from shared import scenario as sc
from shared.store import Store
from tests.fakes import FakeClient, response, text, tool_use, usage


def make_agent(tmp_path, client, session="s1", **kw):
    return StudyAgent(client, tmp_path, session, "2026-11-01", **kw)


def test_loop_runs_tools_until_end_turn_and_batches_results(tmp_path):
    client = FakeClient([
        response(tool_use("a", "add_task", {"course": "intro-cs", "title": "ex2", "due": "2026-11-04", "hours": 2}),
                 tool_use("b", "add_task", {"course": "linear-algebra", "title": "HW3", "due": "2026-11-05", "hours": 3}),
                 stop="tool_use"),
        response(text("הוספתי את שתי המטלות")),
    ])
    out = make_agent(tmp_path, client).turn("add two tasks")
    assert out["model_calls"] == 2 and out["tools"] == ["add_task", "add_task"]
    assert out["response"] == "הוספתי את שתי המטלות"
    last_user = client.requests[1]["messages"][-1]
    assert last_user["role"] == "user"
    assert [b["tool_use_id"] for b in last_user["content"]] == ["a", "b"]  # one message, both results
    assert len(Store(tmp_path).load_tasks()["tasks"]) == 2


def test_tool_error_goes_back_as_is_error(tmp_path):
    client = FakeClient([
        response(tool_use("a", "add_task", {"course": "calculus", "title": "x", "due": "2026-11-04", "hours": 1}),
                 stop="tool_use"),
        response(tool_use("b", "nope", {}), stop="tool_use"),
        response(text("sorry")),
    ])
    make_agent(tmp_path, client).turn("x")
    r1 = client.requests[1]["messages"][-1]["content"][0]
    r2 = client.requests[2]["messages"][-1]["content"][0]
    assert r1["is_error"] and "known:" in r1["content"]
    assert r2["is_error"] and "unknown tool" in r2["content"]


def test_dangling_tool_use_after_max_tokens_gets_a_result(tmp_path):
    client = FakeClient([response(tool_use("a", "list_tasks", {}), stop="max_tokens")])
    agent = make_agent(tmp_path, client)
    agent.turn("x")
    last = agent.history.messages()[-1]
    assert last["role"] == "user" and last["content"][0]["is_error"]


def test_loop_iteration_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "MAX_ITERATIONS", 3)
    client = FakeClient([response(tool_use(str(i), "list_tasks", {}), stop="tool_use") for i in range(10)])
    make_agent(tmp_path, client).turn("x")
    assert len(client.requests) == 3


def test_history_is_append_only_and_reloads_in_new_instance(tmp_path):
    client = FakeClient()
    a = make_agent(tmp_path, client, "sess")
    a.turn("first")
    before = a.history.path.read_text(encoding="utf-8")
    b = make_agent(tmp_path, client, "sess")  # new instance, same session
    b.turn("second")
    after = b.history.path.read_text(encoding="utf-8")
    assert after.startswith(before)
    sent = client.requests[-1]["messages"]
    assert len(sent) == 3 and sent[-1]["content"][-1]["text"] == "second"
    # the earlier messages go back byte-identical (cache markers aside)
    assert sent[0]["content"] == json.loads(before.splitlines()[0])["content"]


def test_memory_reaches_next_session_through_context(tmp_path):
    s1 = FakeClient([response(tool_use("a", "save_memory", {"key": "study_time", "value": "morning"}), stop="tool_use"),
                     response(text("saved"))])
    make_agent(tmp_path, s1, "s1").turn("I like mornings")
    s2 = FakeClient()
    make_agent(tmp_path, s2, "s2").turn("מתי נוח לי ללמוד?")
    req = s2.requests[0]
    assert len(req["messages"]) == 1  # new session: no old conversation
    ctx_block = req["messages"][0]["content"][0]["text"]
    assert "study_time=morning" in ctx_block


def test_context_inlines_only_a_digest(tmp_path):
    from shared.domain import ToolContext, add_task
    ctx = ToolContext(Store(tmp_path), "2026-11-01")
    for i in range(5):
        add_task(ctx, "intro-cs", f"ex{i}", f"2026-11-1{i}", 1)
    client = FakeClient()
    make_agent(tmp_path, client).turn("hi")
    block = client.requests[0]["messages"][0]["content"][0]["text"]
    assert "open_tasks: 5" in block and "(+2 more" in block and "ex4" not in block
    assert "today: 2026-11-01 sunday" in block


def test_caching_breakpoints(tmp_path):
    client = FakeClient([response(tool_use("a", "list_tasks", {}), stop="tool_use"), response(text("ok"))])
    agent = make_agent(tmp_path, client)
    agent.turn("x")
    for req in client.requests:
        assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}
        marked = [b for m in req["messages"] for b in m["content"] if "cache_control" in b]
        assert len(marked) == 1 and marked[0] is req["messages"][-1]["content"][-1]
    assert client.requests[0]["system"] == client.requests[1]["system"]
    assert client.requests[0]["tools"] == client.requests[1]["tools"]
    assert "cache_control" not in agent.history.path.read_text()  # markers never stored


def test_mark_last_does_not_mutate():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
    out = caching.mark_last(msgs)
    assert "cache_control" in out[0]["content"][0] and "cache_control" not in msgs[0]["content"][0]


def test_compaction_over_threshold(tmp_path):
    big = usage(inp=50, read=30_000)
    client = FakeClient([response(text("ok"), u=big),
                         response(text("SUMMARY: tasks T1; study_time=morning")),
                         response(text("ok2"))])
    agent = make_agent(tmp_path, client, compact_at=20_000)
    agent.turn("first")
    out = agent.turn("second")
    assert out["model_calls"] == 2  # summary call + the turn itself
    summarize_req, turn_req = client.requests[1], client.requests[2]
    assert "Summarize" in summarize_req["messages"][-1]["content"][0]["text"]
    assert len(turn_req["messages"]) == 1  # only summary + new turn
    first_block = turn_req["messages"][0]["content"][0]["text"]
    assert "SUMMARY: tasks T1" in first_block
    types = [r["type"] for r in agent.history.records()]
    assert "compaction" in types
    assert agent.history.messages()[0]["content"][0]["text"].startswith("<conversation_summary>")


def test_no_compaction_under_threshold(tmp_path):
    client = FakeClient()
    agent = make_agent(tmp_path, client, compact_at=20_000)
    agent.turn("a")
    agent.turn("b")
    assert len(client.requests) == 2


def test_token_cost():
    u = tokens.TurnUsage()
    u.add(usage(inp=1_000_000, out=0))
    u.add(usage(inp=0, out=1_000_000, read=1_000_000, write=1_000_000))
    assert u.calls == 2
    assert u.cost("claude-opus-5-5") == 4.0 + 20.0 + 0.2 + 5.0


def test_request_shape(tmp_path):
    client = FakeClient()
    make_agent(tmp_path, client).turn("x")
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"] == {"effort": "medium"}
    assert "thinking" not in req and "temperature" not in req
    assert [t["name"] for t in req["tools"]][:2] == ["add_task", "list_tasks"]


def test_full_scenario_offline(tmp_path):
    scen = sc.load(sc.ROOT / "scenario.json")
    logs = run_scenario(FakeClient(), scen, tmp_path, 20_000)
    assert len(logs) == 7
    assert {r["session"] for r in logs} == {"s1", "s2"}
    sessions = list((tmp_path / "sessions").glob("*.jsonl"))
    assert len(sessions) == 2  # session 2 is a new conversation
    res = sc.write_results(tmp_path / "api.json", "api", scen, logs,
                           sc.run_checks(scen, Store(tmp_path), logs), sc.loc_api())
    assert res["summary"]["model_calls"] == 7
    assert res["checks_passed"].endswith("/9")
