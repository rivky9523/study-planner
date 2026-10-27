"""Token usage and cost per model call and per turn."""
from dataclasses import asdict, dataclass

from API.harness import logger

log = logger("tokens.py")

# USD per 1M tokens. cache_write = 5-minute cache write (1.25x input).
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    # Gemini paid tier (the free tier costs $0) - check ai.google.dev/gemini-api/docs/pricing
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cache_read": 0.03, "cache_write": 0.0},
}


def context_tokens(usage) -> int:
    return (usage.input_tokens + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0))


@dataclass
class TurnUsage:
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    calls: int = 0

    def add(self, usage) -> None:
        call = {"input": usage.input_tokens, "cache_read": usage.cache_read_input_tokens or 0,
                "cache_write": usage.cache_creation_input_tokens or 0, "output": usage.output_tokens}
        for k, v in call.items():
            setattr(self, k, getattr(self, k) + v)
        self.calls += 1
        log(f"call {self.calls}: in={call['input']} cache_read={call['cache_read']} "
            f"cache_write={call['cache_write']} out={call['output']}")

    def tokens(self) -> dict:
        d = asdict(self)
        d.pop("calls")
        return d

    def cost(self, model: str) -> float:
        p = PRICES.get(model)
        if p is None:
            log(f"no price for {model}: cost reported as 0")
            return 0.0
        return round(sum(getattr(self, k) * p[k] for k in p) / 1_000_000, 6)
