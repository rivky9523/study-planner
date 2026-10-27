"""Gemini adapter: makes Google's Gemini API look like the Anthropic client the
harness already uses, so loop/tools/history/... run unchanged.

    client = GeminiClient()          # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
    client.beta.messages.create(**anthropic_style_request) -> anthropic-style message

Translation:
  system blocks           -> system_instruction
  tools (JSON schema)     -> function_declarations (parameters_json_schema)
  user/assistant          -> user/model
  text / tool_use         -> text part / function_call part (+ thought_signature)
  tool_result (is_error)  -> function_response {"result": ...} or {"error": ...}
  cache_control, effort, betas, fallbacks -> ignored (Gemini caches implicitly)
Gemini thought signatures are kept on the history blocks (base64) and sent back,
since Gemini requires them on function calls in later requests.
"""
import base64
import re
import time
import uuid
from types import SimpleNamespace

from google.genai import errors, types

from API.harness import logger

log = logger("gemini.py")
DEFAULT_MODEL = "gemini-2.5-flash"
REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}
MAX_RATE_LIMIT_RETRIES = 6  # free tier allows only a few requests per minute


def _retry_delay(err) -> float:
    """Seconds to wait, from Gemini's 'retry in 49.6s' / RetryInfo, else 30."""
    m = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(r"'retryDelay': '(\d+)s'", str(err))
    return float(m.group(1)) + 1 if m else 30.0


class Block(SimpleNamespace):
    """A content block with the attributes the harness reads (type, text, id, name, input)."""

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


def _sig_out(sig: bytes | None) -> str | None:
    return base64.b64encode(sig).decode() if sig else None


def _sig_in(sig: str | None) -> bytes | None:
    return base64.b64decode(sig) if sig else None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content)


def to_contents(messages: list[dict]) -> list[types.Content]:
    names = {}  # tool_use_id -> tool name (function_response needs the name)
    contents = []
    for m in messages:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        parts = []
        for b in blocks:
            if b["type"] == "text":
                parts.append(types.Part(text=b["text"], thought_signature=_sig_in(b.get("thought_signature"))))
            elif b["type"] == "tool_use":
                names[b["id"]] = b["name"]
                call = types.FunctionCall(name=b["name"], args=b["input"], id=b.get("gemini_call_id"))
                parts.append(types.Part(function_call=call, thought_signature=_sig_in(b.get("thought_signature"))))
            elif b["type"] == "tool_result":
                key = "error" if b.get("is_error") else "result"
                orig_id = b["tool_use_id"] if not b["tool_use_id"].startswith("gen_") else None
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name=names.get(b["tool_use_id"], "unknown"), id=orig_id,
                    response={key: _text(b["content"])})))
        contents.append(types.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
    return contents


def to_message(resp) -> SimpleNamespace:
    cand = resp.candidates[0] if resp.candidates else None
    finish = cand.finish_reason.name if cand and cand.finish_reason else "STOP"
    blocks = []
    for p in (cand.content.parts if cand and cand.content and cand.content.parts else []):
        if p.thought:
            continue
        if p.function_call:
            fc = p.function_call
            blocks.append(Block(type="tool_use", id=fc.id or f"gen_{uuid.uuid4().hex[:12]}",
                                name=fc.name, input=dict(fc.args or {}), gemini_call_id=fc.id,
                                thought_signature=_sig_out(p.thought_signature)))
        elif p.text:
            blocks.append(Block(type="text", text=p.text, thought_signature=_sig_out(p.thought_signature)))

    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    elif cand is None or finish in REFUSAL_REASONS:
        stop = "refusal"
    elif finish == "MAX_TOKENS":
        stop = "max_tokens"
    else:
        stop = "end_turn"

    u = resp.usage_metadata
    prompt, cached = (u.prompt_token_count or 0), (u.cached_content_token_count or 0)
    usage = SimpleNamespace(input_tokens=prompt - cached, cache_read_input_tokens=cached,
                            cache_creation_input_tokens=0,
                            output_tokens=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
    log(f"finish={finish} -> stop_reason={stop}")
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=usage)


class GeminiClient:
    def __init__(self, genai_client=None, sleep=time.sleep):
        self._sleep = sleep
        if genai_client is None:
            from google import genai
            genai_client = genai.Client()
        self._g = genai_client
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, *, model, max_tokens, system, tools, messages, **ignored):
        config = types.GenerateContentConfig(
            system_instruction="\n".join(b["text"] for b in system),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"]) for t in tools])],
            max_output_tokens=max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        log(f"generate_content model={model} ({len(messages)} messages; ignored: {sorted(ignored)})")
        contents = to_contents(messages)
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return to_message(self._g.models.generate_content(model=model, contents=contents, config=config))
            except errors.ClientError as e:
                if e.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = _retry_delay(e)
                log(f"rate limited (429), waiting {delay:.0f}s then retrying ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})")
                self._sleep(delay)
