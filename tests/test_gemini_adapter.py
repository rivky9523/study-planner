"""The Gemini adapter, offline: a fake google-genai client, no key."""
from types import SimpleNamespace

from google.genai import types

from API.agent import StudyAgent
from API.providers.gemini import GeminiClient, to_message
from shared.store import Store


def g_response(*parts, finish=types.FinishReason.STOP, prompt=1000, cached=600, out=50, thoughts=30):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, cached_content_token_count=cached,
            candidates_token_count=out, thoughts_token_count=thoughts))


class FakeGenai:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.script.pop(0)


def test_full_turn_through_adapter(tmp_path):
    fake = FakeGenai([
        g_response(types.Part(function_call=types.FunctionCall(name="add_task", args={
            "course": "intro-cs", "title": "ex2", "due": "2026-11-04", "hours": 2}), thought_signature=b"sig1")),
        g_response(types.Part(text="הוספתי")),
    ])
    agent = StudyAgent(GeminiClient(fake), tmp_path, "s1", "2026-11-01", model="gemini-2.5-flash")
    out = agent.turn("add ex2")

    assert out["response"] == "הוספתי" and out["tools"] == ["add_task"] and out["model_calls"] == 2
    assert len(Store(tmp_path).load_tasks()["tasks"]) == 1
    assert out["tokens"] == {"input": 800, "cache_read": 1200, "cache_write": 0, "output": 160}
    assert out["cost_usd"] > 0

    cfg = fake.calls[0]["config"]
    assert cfg.system_instruction.startswith("You are a personal study-planning assistant")
    assert len(cfg.tools[0].function_declarations) == 7

    second = fake.calls[1]["contents"]
    assert [c.role for c in second] == ["user", "model", "user"]
    call_part = second[1].parts[0]
    assert call_part.function_call.name == "add_task" and call_part.thought_signature == b"sig1"
    fr = second[2].parts[0].function_response
    assert fr.name == "add_task" and fr.response["result"].startswith("added T1")


def test_tool_error_becomes_error_response(tmp_path):
    fake = FakeGenai([
        g_response(types.Part(function_call=types.FunctionCall(name="update_task", args={"task_id": "T9"}))),
        g_response(types.Part(text="no such task")),
    ])
    StudyAgent(GeminiClient(fake), tmp_path, "s1", "2026-11-01", model="gemini-2.5-flash").turn("x")
    fr = fake.calls[1]["contents"][2].parts[0].function_response
    assert "error" in fr.response and "no task T9" in fr.response["error"]


def test_history_from_new_instance_is_replayed(tmp_path):
    fake = FakeGenai([g_response(types.Part(text="a")), g_response(types.Part(text="b"))])
    StudyAgent(GeminiClient(fake), tmp_path, "s", "2026-11-01", model="gemini-2.5-flash").turn("one")
    StudyAgent(GeminiClient(fake), tmp_path, "s", "2026-11-01", model="gemini-2.5-flash").turn("two")
    contents = fake.calls[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "a"


def test_stop_reason_mapping():
    assert to_message(g_response(types.Part(text="x"), finish=types.FinishReason.MAX_TOKENS)).stop_reason == "max_tokens"
    assert to_message(g_response(types.Part(text="x"), finish=types.FinishReason.SAFETY)).stop_reason == "refusal"
    assert to_message(g_response(types.Part(text="x"))).stop_reason == "end_turn"
    thought = types.Part(text="thinking...", thought=True)
    msg = to_message(g_response(thought, types.Part(text="answer")))
    assert [b.text for b in msg.content] == ["answer"]


def test_rate_limit_waits_and_retries(tmp_path):
    from google.genai import errors
    err = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                             "message": "Quota exceeded. Please retry in 12.5s."}})
    script = [err, g_response(types.Part(text="ok"))]

    class Flaky(FakeGenai):
        def _gen(self, **kw):
            nxt = self.script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

    slept = []
    client = GeminiClient(Flaky(script), sleep=slept.append)
    out = StudyAgent(client, tmp_path, "s", "2026-11-01", model="gemini-2.5-flash").turn("x")
    assert out["response"] == "ok" and slept == [13.5]


def test_other_errors_are_raised(tmp_path):
    import pytest
    from google.genai import errors

    class Broken(FakeGenai):
        def _gen(self, **kw):
            raise errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(errors.ClientError):
        StudyAgent(GeminiClient(Broken([]), sleep=lambda s: None), tmp_path, "s", "2026-11-01",
                   model="gemini-2.5-flash").turn("x")
