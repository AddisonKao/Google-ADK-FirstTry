"""
Golden dataset eval runner.
Usage:
    python eval/run_eval.py [--threshold 7.0] [--max-regression 0.10]
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from langfuse import Langfuse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.otel_setup import setup as setup_otel
setup_otel()

from agent.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from eval.judge import judge

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
SCORE_THRESHOLD = float(os.getenv("EVAL_SCORE_THRESHOLD", "7.0"))
MAX_REGRESSION = float(os.getenv("EVAL_MAX_REGRESSION", "0.10"))


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_dataset() -> list[dict]:
    cases = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def run_agent(text: str) -> tuple[str, list[str]]:
    """Run agent on input text. Returns (response_text, tools_called)."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name="kafka_agent_eval",
    )
    session = await session_service.create_session(
        app_name="kafka_agent_eval", user_id="eval-user"
    )
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=text)]
    )
    response_text = ""
    tools_called = []
    async for event in runner.run_async(
        user_id="eval-user",
        session_id=session.id,
        new_message=content,
    ):
        if hasattr(event, "tool_name") and event.tool_name:
            tools_called.append(event.tool_name)
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    return response_text, tools_called


def get_baseline_score(lf: Langfuse) -> float | None:
    """Fetch mean score from the last successful eval run in Langfuse."""
    try:
        runs = lf.get_dataset_runs(dataset_name="golden-dataset")
        passing = [r for r in runs.data if r.metadata and r.metadata.get("passed")]
        if not passing:
            return None
        last = sorted(passing, key=lambda r: r.created_at, reverse=True)[0]
        return last.metadata.get("mean_score")
    except Exception as e:
        print(f"[eval] Could not fetch baseline: {e}")
        return None


def main():
    lf = Langfuse(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3030"),
    )

    git_sha = get_git_sha()
    run_name = f"eval-{git_sha}"
    cases = load_dataset()
    print(f"[eval] Running {len(cases)} cases (run: {run_name})")

    baseline = get_baseline_score(lf)
    if baseline is not None:
        print(f"[eval] Baseline mean score: {baseline:.2f}")

    scores = []
    results = []

    for case in cases:
        case_id = case["id"]
        print(f"[eval] Case {case_id}: {case['input'][:60]}...")

        response, tools_called = asyncio.run(run_agent(case["input"]))
        verdict = judge(
            input_text=case["input"],
            response=response,
            expected_criteria=case["expected_criteria"],
            expected_tool=case.get("expected_tool"),
            tools_called=tools_called,
        )

        scores.append(verdict["score"])
        results.append({**case, "response": response, "verdict": verdict})
        status = "✓" if verdict["pass"] else "✗"
        print(f"  {status} score={verdict['score']} — {verdict['reason']}")

        # Log to Langfuse
        trace = lf.trace(name=f"{run_name}/{case_id}", input=case["input"], output=response)
        lf.score(
            trace_id=trace.id,
            name="llm-judge",
            value=verdict["score"],
            comment=verdict["reason"],
        )

    mean_score = sum(scores) / len(scores)
    pass_rate = sum(1 for r in results if r["verdict"]["pass"]) / len(results)
    passed = mean_score >= SCORE_THRESHOLD

    print(f"\n[eval] Results: mean={mean_score:.2f} pass_rate={pass_rate:.0%} threshold={SCORE_THRESHOLD}")

    # Regression check
    regression_fail = False
    if baseline is not None:
        regression = (baseline - mean_score) / baseline
        if regression > MAX_REGRESSION:
            print(f"[eval] REGRESSION: score dropped {regression:.1%} vs baseline {baseline:.2f} (max allowed: {MAX_REGRESSION:.0%})")
            regression_fail = True
        else:
            print(f"[eval] Regression check OK: {regression:.1%} drop vs baseline")

    overall_pass = passed and not regression_fail

    # Post experiment run to Langfuse
    langfuse_url = os.getenv("LANGFUSE_HOST", "http://localhost:3030")
    lf.trace(
        name=run_name,
        metadata={
            "passed": overall_pass,
            "mean_score": mean_score,
            "pass_rate": pass_rate,
            "baseline": baseline,
            "git_sha": git_sha,
            "threshold": SCORE_THRESHOLD,
        },
    )
    lf.flush()

    print(f"\n[eval] {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print(f"[eval] Langfuse: {langfuse_url}")

    # Write summary for CI comment
    summary = {
        "run_name": run_name,
        "mean_score": round(mean_score, 2),
        "pass_rate": round(pass_rate, 2),
        "passed": overall_pass,
        "langfuse_url": langfuse_url,
        "baseline": baseline,
    }
    with open("eval_summary.json", "w") as f:
        json.dump(summary, f)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
