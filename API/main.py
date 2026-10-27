"""Run the scenario on the raw Messages API version.

    python API/main.py --scenario scenario.json --out results/api.json
Needs ANTHROPIC_API_KEY in the environment (never in code or git).

Optional, for development with a Gemini key (not the graded comparison):
    python API/main.py --provider gemini --out results/api-gemini.json
Needs GEMINI_API_KEY in the environment.
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import scenario as sc  # noqa: E402


def run(client, scen: dict, state_dir, compact_at: int) -> list[dict]:
    from API.agent import StudyAgent
    logs = []
    for session in scen["sessions"]:
        # New session = new agent instance; everything it knows comes from disk.
        agent = StudyAgent(client, state_dir, f"{session['name']}-{uuid.uuid4().hex[:6]}", scen["today"],
                           scen["model"], scen["effort"], compact_at)
        for t in session["turns"]:
            print(f"\n=== [{session['name']}/{t['id']}] user: {t['user']}", flush=True)
            log = agent.turn(t["user"])
            print(f"=== assistant: {log['response']}\n", flush=True)
            logs.append({"session": session["name"], "turn": t["id"], "user": t["user"], **log})
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenario.json")
    ap.add_argument("--out", default=None, help="default results/api.json (gemini: results/api-gemini.json)")
    ap.add_argument("--state-dir", default="state/api")
    ap.add_argument("--compact-at", type=int, default=20_000, help="context tokens that trigger compaction")
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--model", default=None, help="override the scenario model (gemini default: gemini-2.5-flash)")
    args = ap.parse_args()
    scen = sc.load(args.scenario)

    if args.provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("Set GEMINI_API_KEY in your terminal first (export GEMINI_API_KEY=...).")
        from google import genai
        from API.providers.gemini import DEFAULT_MODEL, GeminiClient
        client, scen["model"] = GeminiClient(), args.model or DEFAULT_MODEL
        versions = {"google_genai_version": genai.__version__}
        out = args.out or "results/api-gemini.json"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY in your terminal first (export ANTHROPIC_API_KEY=...).")
        import anthropic
        client, scen["model"] = anthropic.Anthropic(), args.model or scen["model"]
        versions = {"anthropic_version": anthropic.__version__}
        out = args.out or "results/api.json"

    state = sc.clean_state(args.state_dir)
    logs = run(client, scen, state, args.compact_at)
    from shared.store import Store
    checks = sc.run_checks(scen, Store(state), logs)
    res = sc.write_results(out, "api", scen, logs, checks, sc.loc_api(),
                           {"provider": args.provider, "compact_at": args.compact_at, **versions})
    print(f"checks {res['checks_passed']} | {res['summary']}")


if __name__ == "__main__":
    main()
