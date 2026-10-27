"""Run the scenario on the Claude Agent SDK version.

    python SDK/main.py --scenario scenario.json --out results/sdk.json
Runs on your Claude Code login. ANTHROPIC_API_KEY must NOT be set, otherwise
billing moves to the key.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import scenario as sc  # noqa: E402


async def run(scen: dict, state_dir) -> list[dict]:
    from SDK.agent import StudyAgentSDK
    logs = []
    for session in scen["sessions"]:
        agent = StudyAgentSDK(state_dir, scen["today"], scen["model"], scen["effort"])  # new instance per session
        for t in session["turns"]:
            print(f"\n=== [{session['name']}/{t['id']}] user: {t['user']}", flush=True)
            log = await agent.turn(t["user"])
            print(f"[SDK] tools={log['tools']} calls={log['model_calls']} cost=${log['cost_usd']}")
            print(f"=== assistant: {log['response']}\n", flush=True)
            logs.append({"session": session["name"], "turn": t["id"], "user": t["user"], **log})
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenario.json")
    ap.add_argument("--out", default="results/sdk.json")
    ap.add_argument("--state-dir", default="state/sdk")
    args = ap.parse_args()

    if os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is set - unset it so the SDK runs on your Claude Code login.")
    import claude_agent_sdk

    scen = sc.load(args.scenario)
    state = sc.clean_state(args.state_dir)
    logs = asyncio.run(run(scen, state))
    from shared.store import Store
    checks = sc.run_checks(scen, Store(state), logs)
    res = sc.write_results(args.out, "sdk", scen, logs, checks, sc.loc_sdk(),
                           {"claude_agent_sdk_version": claude_agent_sdk.__version__})
    print(f"checks {res['checks_passed']} | {res['summary']}")


if __name__ == "__main__":
    main()
