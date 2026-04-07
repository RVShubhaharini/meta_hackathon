#!/usr/bin/env python3
"""
Baseline inference script for AI Content Moderation OpenEnv.

Usage:
    python inference.py [--task task_easy|task_medium|task_hard] [--output results.json]

Environment variables required:
    API_BASE_URL   – OpenAI-compatible base URL (e.g. https://api.anthropic.com/v1)
    MODEL_NAME     – Model identifier (e.g. claude-sonnet-4-20250514)
    OPENAI_API_KEY – API key

Output format:
    [START] ... [STEP] ... [END]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from environment.env import ModerationEnv
from environment.models import Action, ContentCategory, ModerationAction, Severity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.anthropic.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "claude-sonnet-4-20250514")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are an expert AI content moderation system for a social media platform.

Your task is to analyze social media posts and make moderation decisions.

For each post you will be given:
- The post content (may be in English, Hindi, or Tamil)
- User history and reputation score
- Thread/conversation context (if available)
- Post metadata

You must respond with ONLY a valid JSON object following this exact schema:
{
  "post_id": "<post_id from input>",
  "classification": "<one of: safe, spam, hate_speech, harassment, violence, misinformation, sexual, self_harm>",
  "severity": "<one of: none, low, medium, high>",
  "action": "<one of: allow, flag, remove, escalate>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}

Guidelines:
- safe + none severity → allow
- spam/mild hate → flag or remove based on severity  
- high severity harmful content → remove
- ambiguous or borderline cases (especially mental health) → escalate
- Consider user reputation: low reputation users get less benefit of the doubt
- Be consistent: treat identical content the same regardless of identity groups mentioned
- For thread context: consider the full conversation before deciding

Severity guide:
- none: completely safe content
- low: mildly problematic, could be interpreted either way
- medium: clearly problematic but not extreme
- high: severely harmful, threats, targeted harassment, crisis situations

Do not add any text outside the JSON object."""

USER_PROMPT_TEMPLATE = """Moderate this social media post:

POST ID: {post_id}
CONTENT: {content}
LANGUAGE: {language}
USER REPUTATION SCORE: {reputation:.2f} (0=worst, 1=best)
USER HISTORY: {flagged} flagged posts, {removed} removed posts out of {total} total
PRIOR VIOLATIONS: {violations}

THREAD CONTEXT:
{thread_context}

METADATA:
- Reports received: {reports}
- Is reply: {is_reply}
- Hashtags: {hashtags}

Respond with the JSON moderation decision only."""


def format_thread_context(obs) -> str:
    ctx = obs.context
    if not ctx.thread_posts:
        return "No thread context (standalone post)"
    lines = [f"Thread topic: {ctx.thread_topic or 'N/A'}"]
    for i, p in enumerate(ctx.thread_posts, 1):
        lines.append(f"  [{i}] {p}")
    return "\n".join(lines)


def build_prompt(obs) -> str:
    return USER_PROMPT_TEMPLATE.format(
        post_id=obs.post_id,
        content=obs.content,
        language=obs.language,
        reputation=obs.user_reputation_score,
        flagged=obs.user_history.flagged_count,
        removed=obs.user_history.removed_count,
        total=obs.user_history.total_posts,
        violations=", ".join(obs.user_history.prior_violations) or "none",
        thread_context=format_thread_context(obs),
        reports=obs.metadata.reports,
        is_reply=obs.metadata.is_reply,
        hashtags=", ".join(obs.metadata.hashtags) or "none",
    )


def parse_action(response_text: str, post_id: str) -> Optional[Action]:
    """Parse LLM response into an Action object."""
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return Action(
            post_id=post_id,
            classification=ContentCategory(data["classification"]),
            severity=Severity(data["severity"]),
            action=ModerationAction(data["action"]),
            confidence=float(data.get("confidence", 1.0)),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as e:
        print(f"  [WARN] Failed to parse action for {post_id}: {e}", file=sys.stderr)
        # Fallback: safe default
        return Action(
            post_id=post_id,
            classification=ContentCategory.SAFE,
            severity=Severity.NONE,
            action=ModerationAction.ALLOW,
            confidence=0.0,
            reasoning=f"Parse error: {e}",
        )


def log(tag: str, data: dict) -> None:
    """Structured log line."""
    print(f"[{tag}] {json.dumps(data)}", flush=True)


def run_inference(task_id: str, output_path: Optional[str] = None) -> dict:
    """
    Run a full episode on the given task using the LLM agent.

    Returns a results dict with per-step details and aggregate scores.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    env = ModerationEnv(task_id=task_id)
    obs = env.reset()

    start_ts = datetime.now(timezone.utc).isoformat()
    log("START", {
        "task_id": task_id,
        "model": MODEL_NAME,
        "timestamp": start_ts,
        "total_posts": len(env._task_records),
    })

    results = []
    total_reward = 0.0
    step = 0

    while True:
        step += 1
        post_id = obs.post_id
        if post_id == "TERMINAL":
            break

        prompt = build_prompt(obs)
        t0 = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # deterministic
            max_tokens=400,
        )
        elapsed = round(time.time() - t0, 3)
        raw_response = response.choices[0].message.content

        action = parse_action(raw_response, post_id)
        obs_next, reward, done, info = env.step(action)

        total_reward += reward

        step_data = {
            "step": step,
            "post_id": post_id,
            "action": action.model_dump(),
            "reward": reward,
            "ground_truth_label": info["ground_truth_label"],
            "ground_truth_action": info["ground_truth_action"],
            "reward_breakdown": info["reward_breakdown"],
            "elapsed_sec": elapsed,
        }
        log("STEP", step_data)
        results.append(step_data)

        if done:
            break
        obs = obs_next

    state = env.state()
    avg_reward = total_reward / max(len(results), 1)

    summary = {
        "task_id": task_id,
        "model": MODEL_NAME,
        "timestamp_start": start_ts,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
        "total_steps": state.current_step,
        "average_reward": round(avg_reward, 4),
        "cumulative_reward": round(state.cumulative_reward, 4),
        "correct_classifications": state.correct_classifications,
        "false_positives": state.false_positives,
        "false_negatives": state.false_negatives,
        "escalations_correct": state.escalations_correct,
        "bias_violations": state.bias_violations,
        "steps": results,
    }

    log("END", {k: v for k, v in summary.items() if k != "steps"})

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_path}", file=sys.stderr)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference for AI Content Moderation OpenEnv"
    )
    parser.add_argument(
        "--task",
        choices=["task_easy", "task_medium", "task_hard", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    tasks = (
        ["task_easy", "task_medium", "task_hard"]
        if args.task == "all"
        else [args.task]
    )

    all_results = {}
    for task_id in tasks:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running task: {task_id}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        result = run_inference(
            task_id=task_id,
            output_path=args.output.replace(".json", f"_{task_id}.json")
            if args.output
            else None,
        )
        all_results[task_id] = result["average_reward"]

    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES:", file=sys.stderr)
    for tid, score in all_results.items():
        print(f"  {tid}: {score:.4f}", file=sys.stderr)
    overall = sum(all_results.values()) / len(all_results)
    print(f"  OVERALL: {overall:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
