"""A fake Anthropic client: scripted responses, records every request. No key needed."""
from types import SimpleNamespace


class Block(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def text(t):
    return Block(type="text", text=t)


def tool_use(id_, name, input_):
    return Block(type="tool_use", id=id_, name=name, input=input_)


def usage(inp=100, out=20, read=0, write=0):
    return SimpleNamespace(input_tokens=inp, output_tokens=out,
                           cache_read_input_tokens=read, cache_creation_input_tokens=write)


def response(*blocks, stop="end_turn", u=None):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=u or usage())


class FakeClient:
    """script: list of responses (or callables req -> response). When it runs out,
    answers end_turn "ok" so long scenarios still finish."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **req):
        self.requests.append(req)
        if not self.script:
            return response(text("ok"))
        nxt = self.script.pop(0)
        return nxt(req) if callable(nxt) else nxt
