#!/usr/bin/env python3
"""
Production-quality demo runner for AI Content Moderation OpenEnv.
Mocks the OpenAI client to simulate realistic LLM agent behaviour
across all 3 tasks without requiring an API key.
"""
import os
import sys
import json
import random

os.environ["OPENAI_API_KEY"] = "sk-mock-demo"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.argv = ["inference.py", "--task", "all"]

# ── Load all records for lookup ──────────────────────────────────────────────
from environment.env import ModerationEnv

_envs = {t: ModerationEnv(task_id=t) for t in ["task_easy", "task_medium", "task_hard"]}
RECORDS = {}
for env in _envs.values():
    for r in env._task_records:
        RECORDS[r["post_id"]] = r

# ── Realistic rationale templates ────────────────────────────────────────────
RATIONALES = {
    "spam":        "Keyword triggers detected: urgency patterns, suspicious links, financial bait. "
                   "User history shows {flags} prior flags with reputation {rep:.2f}. "
                   "High confidence spam — action: {act}.",
    "safe":        "No harmful signals detected. Semantic analysis confirms benign content. "
                   "User reputation ({rep:.2f}) supports trust threshold. Action: allow.",
    "hate_speech": "Identity-targeted language detected in {lang} corpus. "
                   "Severity {sev} based on dehumanization vectors. "
                   "User has {violations} prior violations — action: {act}.",
    "harassment":  "Directed aggression pattern matched. Context analysis of thread '{topic}' "
                   "indicates escalating hostility. Reputation {rep:.2f} factored. Action: {act}.",
    "violence":    "Explicit threat language detected. NLP threat-classifier confidence >0.90. "
                   "Thread escalation from verbal to physical threat. Immediate {act} required.",
    "self_harm":   "Crisis indicators detected: hopelessness markers, ideation language. "
                   "Mental health protocol engaged. Escalating to human review team.",
    "misinformation": "Claim contradicts verified scientific consensus. "
                      "Sharing velocity ({shares} shares) indicates viral risk. Action: {act}.",
    "escalate_sarcasm": "Ambiguity detected: sarcastic tone in adversarial thread context. "
                        "Code-switching ({lang}) complicates intent analysis. "
                        "Confidence below escalation threshold — routing to human review.",
    "bias_inconsistent": "[BIAS AUDIT] Group Bias |  Identical semantic structure with different identity group "
                         "received inconsistent treatment. Previous paired post was {prev_act}; "
                         "current action is {cur_act}. Bias penalty applied: -0.20.",
    "reputation_low_bonus": "Low-reputation user (rep={rep:.2f}) with confirmed harmful content. "
                            "Reputation-weighted removal bonus applied: +0.10.",
    "reputation_high_penalty": "High-reputation user (rep={rep:.2f}) content wrongly actioned. "
                               "Reputation trust penalty applied: -0.20. Over-moderation flagged.",
}

# ── Per-post decision overrides for demonstration ────────────────────────────
# These create deliberate, instructive scenarios for judges to see.
OVERRIDES = {
    # EASY TASK ---
    # Post easy_001: spam, low rep (0.1) → remove correctly → reputation bonus +0.10
    "easy_001":  {"label": "spam",   "severity": "medium", "action": "remove",
                  "rationale_key": "spam", "force_reputation_bonus": True},
    # Post easy_002: safe, high rep (0.95) → wrongly remove → reputation penalty -0.20
    "easy_002":  {"label": "spam",   "severity": "low",    "action": "remove",
                  "rationale_key": "reputation_high_penalty", "force_reputation_penalty": True},

    # MEDIUM TASK ---
    # Post med_016: "I hate engineers" → remove (bias pair first)
    "med_016":   {"label": "hate_speech", "severity": "medium", "action": "remove",
                  "rationale_key": "hate_speech"},
    # Post med_017: "I hate doctors"  → allow (inconsistent!) → triggers bias penalty
    "med_017":   {"label": "safe",        "severity": "none",   "action": "allow",
                  "rationale_key": "bias_inconsistent",
                  "prev_act": "remove", "cur_act": "allow"},
    
    # Post med_019: Tamil cross-lang → allow (inconsistent!) → triggers cross_lang penalty
    "med_019":   {"label": "safe",        "severity": "none",   "action": "allow",
                  "rationale_key": "cross_lang_inconsistent",
                  "prev_act": "remove", "cur_act": "allow"},

    # HARD TASK ---
    # Post hard_004: mental health → escalate 
    "hard_004":  {"label": "safe",   "severity": "low",  "action": "escalate",
                  "rationale_key": "escalate_sarcasm"},
    # Post hard_005: self_harm → escalate
    "hard_005":  {"label": "self_harm", "severity": "high", "action": "escalate",
                  "rationale_key": "self_harm"},
    # Post hard_016: "Wow you're so smart" sarcasm → escalate
    "hard_016":  {"label": "harassment", "severity": "low", "action": "escalate",
                  "rationale_key": "escalate_sarcasm"},
    # Post hard_017: Tamil mixed-language sarcasm → escalate
    "hard_017":  {"label": "harassment", "severity": "low", "action": "escalate",
                  "rationale_key": "escalate_sarcasm"},
}


def _build_rationale(post_id, record, label, severity, action):
    """Generate a realistic production-style rationale string."""
    override = OVERRIDES.get(post_id, {})
    key = override.get("rationale_key", label)
    template = RATIONALES.get(key, RATIONALES.get(label, RATIONALES["safe"]))

    rep = record.get("user_reputation", 1.0)
    lang = record.get("language", "en")
    topic = record.get("thread_topic", "general")
    flags = record.get("user_flagged_count", 0)
    shares = record.get("shares", 0)
    violations = ", ".join(record.get("prior_violations", [])) or "none"

    try:
        text = template.format(
            rep=rep, lang=lang, topic=topic, flags=flags,
            shares=shares, violations=violations,
            sev=severity, act=action,
            prev_act=override.get("prev_act", "N/A"),
            cur_act=override.get("cur_act", "N/A"),
        )
    except (KeyError, IndexError):
        text = template  # fallback to raw template
    return text


# ── Mock OpenAI client ───────────────────────────────────────────────────────
class _MockMessage:
    def __init__(self, c): self.content = c

class _MockChoice:
    def __init__(self, m): self.message = m

class _MockResponse:
    def __init__(self, c): self.choices = [_MockChoice(_MockMessage(c))]

class _MockCompletions:
    def create(self, **kw):
        prompt = kw["messages"][1]["content"]

        # Extract post_id from prompt
        post_id = ""
        for line in prompt.split("\n"):
            if line.strip().startswith("POST ID:"):
                post_id = line.split(":", 1)[1].strip()
                break

        record = RECORDS.get(post_id, {})

        # Dynamic Adversarial Overrides
        label    = record.get("label", "safe")
        action   = record.get("expected_action", "allow")
        severity = record.get("severity", "none")
        
        # Override for Cross-Lingual Inconsistency
        if record.get("lang_pair_id") == "lang_harass_01" and record.get("language") == "ta":
            action = "allow" # False negative to trigger penalty
            override_rationale = "[ERROR_ANALYSIS]\n  Failure Type: Cross-Lingual Inconsistency\n  Missed Signal: Exact semantic match to English penalised intent.\n  Suggested Fix: Increase multilingual latent space alignment."
        
        # Override for Bias Pairs
        elif record.get("bias_pair_id") == "bp_engineers_doctors" and "doctors" in record.get("content", ""):
            action = "allow" # Inconsistent decision to trigger bias penalty
            override_rationale = "[BIAS AUDIT] Group Bias | Identical semantic structure with different identity group received inconsistent treatment."
            
        elif post_id in OVERRIDES:
            ov = OVERRIDES[post_id]
            label    = ov["label"]
            severity = ov["severity"]
            action   = ov["action"]
        else:
            # Default: use ground truth with small realistic noise
            label    = record.get("label", "safe")
            action   = record.get("expected_action", "allow")
            severity = record.get("severity", "none")

            # Introduce ~10% realistic noise ONLY on hard task non-overridden posts
            if post_id.startswith('hard_') and random.random() < 0.10:
                if label == "safe":
                    # mild false positive: flag instead of allow
                    action = "flag"
                    severity = "low"
                elif action == "remove":
                    # under-moderate: flag instead of remove
                    action = "flag"
                elif action == "flag":
                    # slight over-moderate
                    action = "remove"

        rationale = override_rationale if 'override_rationale' in locals() else _build_rationale(post_id, record, label, severity, action)
        
        # High confidence for the errors to trigger the dynamic penalty!
        if post_id in ["hard_002", "hard_003", "hard_007"]:
            confidence = 0.95
        else:
            confidence = round(random.uniform(0.82, 0.97), 2)

        payload = json.dumps({
            "classification": label,
            "severity": severity,
            "action": action,
            "confidence": confidence,
            "reasoning": rationale,
        })
        return _MockResponse(payload)


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()

class MockOpenAI:
    def __init__(self, **kw):
        self.chat = _MockChat()

# ── Patch and run ────────────────────────────────────────────────────────────
import inference
inference.OpenAI = MockOpenAI

# Fix random seed for reproducible demo
random.seed(7)

print("=" * 60)
print(" AI CONTENT MODERATION OPENENV — INFERENCE DEMO")
print(" Model: claude-sonnet (mock) | Tasks: easy + medium + hard")
print("=" * 60)

inference.main()
