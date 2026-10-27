"""Fill the tables in REPORT.md from results/api.json and results/sdk.json.

    python scripts/make_report.py
Replaces everything between the TABLES markers; the written analysis is untouched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.scenario import HARNESS_PARTS  # noqa: E402

START, END = "<!-- TABLES:START -->", "<!-- TABLES:END -->"


def load(name):
    p = ROOT / "results" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tables(api, sdk) -> str:
    def s(r, path):
        if r is None:
            return "-"
        v = r
        for k in path:
            v = v[k]
        return f"{v:,}" if isinstance(v, int) else str(v)

    rows = [("input (uncached)", ["summary", "tokens", "input"]),
            ("cache read", ["summary", "tokens", "cache_read"]),
            ("cache write", ["summary", "tokens", "cache_write"]),
            ("output", ["summary", "tokens", "output"]),
            ("% cache (read / all input)", ["summary", "cache_hit_pct"]),
            ("model calls", ["summary", "model_calls"]),
            ("cost (USD)", ["summary", "cost_usd"]),
            ("time (s)", ["summary", "seconds"]),
            ("checks passed", ["checks_passed"])]
    out = ["### API מול SDK", "", "| | API | SDK |", "|---|---:|---:|"]
    out += [f"| {label} | {s(api, p)} | {s(sdk, p)} |" for label, p in rows]

    out += ["", "### לפי תור", "", "| תור | API in/read/write/out | API calls | API $ | SDK in/read/write/out | SDK calls | SDK $ |",
            "|---|---|---:|---:|---|---:|---:|"]
    n = max(len(api["turns"]) if api else 0, len(sdk["turns"]) if sdk else 0)
    for i in range(n):
        cells = []
        for r in (api, sdk):
            if r and i < len(r["turns"]):
                t = r["turns"][i]
                k = t["tokens"]
                cells += [f"{k['input']}/{k['cache_read']}/{k['cache_write']}/{k['output']}",
                          str(t["model_calls"]), f"{t['cost_usd']:.4f}"]
            else:
                cells += ["-", "-", "-"]
        turn = (api or sdk)["turns"][i]
        out.append(f"| {turn['session']}/{turn['turn']} | " + " | ".join(cells) + " |")

    out += ["", "### שורות קוד ל-harness", "", "| חלק | API (`API/harness/*.py`) | SDK (שורות `# replaces:`) |",
            "|---|---:|---:|"]
    for part in HARNESS_PARTS:
        a = api["harness_loc"][part] if api else "-"
        b = sdk["harness_loc"][part] if sdk else "-"
        out.append(f"| {part} | {a} | {b if b else '0 (אוטומטי)' if part in ('compaction', 'caching') else b} |")
    if api and sdk:
        out.append(f"| **סה\"כ** | **{sum(api['harness_loc'].values())}** | **{sum(sdk['harness_loc'].values())}** |")

    out += ["", "### בדיקות", "", "| בדיקה | API | SDK |", "|---|:-:|:-:|"]
    checks = (api or sdk)["checks"] if (api or sdk) else []
    for i, c in enumerate(checks):
        mark = lambda r: ("✅" if r["checks"][i]["pass"] else "❌") if r else "-"
        out.append(f"| {c['id']}: {c['description']} | {mark(api)} | {mark(sdk)} |")
    return "\n".join(out)


def main():
    api, sdk = load("api"), load("sdk")
    if not (api or sdk):
        sys.exit("no results yet - run API/main.py and SDK/main.py first")
    report = ROOT / "REPORT.md"
    text = report.read_text(encoding="utf-8")
    a, b = text.index(START) + len(START), text.index(END)
    report.write_text(text[:a] + "\n" + tables(api, sdk) + "\n" + text[b:], encoding="utf-8")
    print("REPORT.md tables updated")


if __name__ == "__main__":
    main()
