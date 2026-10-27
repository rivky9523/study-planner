import asyncio

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

import SDK.agent as sdk_agent
from SDK.tools import _wrap, allowed_tool_names
from shared.domain import TOOLS_BY_NAME, ToolContext, save_memory
from shared.store import Store


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(Store(tmp_path), "2026-11-01")


def test_tool_wrapper_ok_and_error(ctx):
    t = _wrap(ctx, TOOLS_BY_NAME["add_task"])
    ok = asyncio.run(t.handler({"course": "intro-cs", "title": "ex2", "due": "2026-11-04", "hours": 2}))
    assert ok["content"][0]["text"].startswith("added T1") and "is_error" not in ok
    bad = asyncio.run(t.handler({"course": "nope", "title": "x", "due": "2026-11-04", "hours": 2}))
    assert bad["is_error"]


def test_options(tmp_path):
    a = sdk_agent.StudyAgentSDK(tmp_path, "2026-11-01")
    o = a.options()
    assert o.strict_mcp_config is True
    assert o.tools == []
    assert o.allowed_tools == allowed_tool_names() and len(o.allowed_tools) == 7
    assert all(n.startswith("mcp__study__") for n in o.allowed_tools)
    assert o.system_prompt.startswith("You are a personal study-planning assistant")
    assert o.setting_sources == ["project"] and o.cwd == str(tmp_path)
    assert o.model == "claude-opus-5-5" and o.effort == "medium"
    assert o.resume is None


def test_claude_md_carries_memory(tmp_path, ctx):
    save_memory(ctx, "study_time", "morning")
    sdk_agent.StudyAgentSDK(tmp_path, "2026-11-01")
    md = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "- study_time: morning (preference)" in md


class FakeSDKClient:
    def __init__(self, options):
        FakeSDKClient.last_options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def query(self, prompt):
        FakeSDKClient.prompt = prompt

    async def receive_response(self):
        yield AssistantMessage(content=[ToolUseBlock("1", "mcp__study__list_tasks", {})], model="m", message_id="m1")
        yield AssistantMessage(content=[TextBlock("done")], model="m", message_id="m2")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=2,
                            session_id="sess-1", total_cost_usd=0.01, result="done",
                            usage={"input_tokens": 10, "cache_read_input_tokens": 5000,
                                   "cache_creation_input_tokens": 200, "output_tokens": 30})


def test_turn_collects_usage_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk_agent, "ClaudeSDKClient", FakeSDKClient)
    a = sdk_agent.StudyAgentSDK(tmp_path, "2026-11-01")
    out = asyncio.run(a.turn("מה נשאר?"))
    assert out["tools"] == ["list_tasks"] and out["model_calls"] == 2
    assert out["tokens"] == {"input": 10, "cache_read": 5000, "cache_write": 200, "output": 30}
    assert out["cost_usd"] == 0.01 and out["response"] == "done"
    assert FakeSDKClient.prompt.startswith("(today: 2026-11-01 sunday)")
    asyncio.run(a.turn("again"))
    assert FakeSDKClient.last_options.resume == "sess-1"
