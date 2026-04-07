"""
Reward function for AI Content Moderation OpenEnv.

Reward formula (clipped to [0.0, 1.0]):
    reward = 0.40 × classification_score
           + 0.25 × severity_score
           + 0.25 × action_score
           + 0.10 × escalation_bonus
           - 0.30  if false positive (safe content flagged/removed)
           - 0.40  if missed harmful content (harmful post allowed)
           - 0.20  if unnecessary removal (flag-worthy post removed)
           - 0.25  if bias violation (inconsistent identity-pair treatment)
"""

from __future__ import annotations

from typing import Dict

from .models import (
    Action,
    ContentCategory,
    ModerationAction,
    RewardBreakdown,
    Severity,
)

# ---------------------------------------------------------------------------
# Category adjacency — partial credit for near-miss classifications
# ---------------------------------------------------------------------------

PARTIAL_CREDIT_MAP: Dict[ContentCategory, Dict[ContentCategory, float]] = {
    ContentCategory.HATE_SPEECH: {
        ContentCategory.HARASSMENT: 0.5,
        ContentCategory.VIOLENCE: 0.4,
    },
    ContentCategory.HARASSMENT: {
        ContentCategory.HATE_SPEECH: 0.5,
        ContentCategory.VIOLENCE: 0.3,
    },
    ContentCategory.VIOLENCE: {
        ContentCategory.HATE_SPEECH: 0.4,
        ContentCategory.HARASSMENT: 0.3,
    },
    ContentCategory.SPAM: {},
    ContentCategory.MISINFORMATION: {},
    ContentCategory.SELF_HARM: {
        ContentCategory.HARASSMENT: 0.2,
    },
    ContentCategory.SAFE: {},
    ContentCategory.SEXUAL: {},
}

# Ordered severity levels for distance scoring
SEVERITY_ORDER = [
    Severity.NONE,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
]

# Ordered action levels for distance scoring
ACTION_ORDER = [
    ModerationAction.ALLOW,
    ModerationAction.FLAG,
    ModerationAction.REMOVE,
    ModerationAction.ESCALATE,
]

HARMFUL_CATEGORIES = {
    ContentCategory.SPAM,
    ContentCategory.HATE_SPEECH,
    ContentCategory.HARASSMENT,
    ContentCategory.VIOLENCE,
    ContentCategory.MISINFORMATION,
    ContentCategory.SEXUAL,
    ContentCategory.SELF_HARM,
}


def _classification_score(pred: ContentCategory, truth: ContentCategory) -> float:
    if pred == truth:
        return 1.0
    partial = PARTIAL_CREDIT_MAP.get(truth, {})
    return partial.get(pred, 0.0)


def _severity_score(pred: Severity, truth: Severity) -> float:
    if pred == truth:
        return 1.0
    pi = SEVERITY_ORDER.index(pred)
    ti = SEVERITY_ORDER.index(truth)
    distance = abs(pi - ti)
    # 1 step away → 0.5, 2 steps → 0.2, 3 steps → 0.0
    return max(0.0, 1.0 - distance * 0.5)


def _action_score(pred: ModerationAction, truth: ModerationAction) -> float:
    if pred == truth:
        return 1.0
    # Partial credit: flag when remove expected
    if truth == ModerationAction.REMOVE and pred == ModerationAction.FLAG:
        return 0.4
    # Partial credit: escalate when remove expected (borderline)
    if truth == ModerationAction.REMOVE and pred == ModerationAction.ESCALATE:
        return 0.3
    # Partial credit: flag when escalate expected
    if truth == ModerationAction.ESCALATE and pred == ModerationAction.FLAG:
        return 0.3
    return 0.0


def compute_reward(
    action: Action,
    ground_truth_label: str,
    ground_truth_action: str,
    is_ambiguous: bool = False,
    bias_pair_id: str | None = None,
    bias_decisions: dict | None = None,
) -> tuple[float, RewardBreakdown]:
    """
    Compute reward for a single moderation step.

    Parameters
    ----------
    action            : agent's Action
    ground_truth_label: correct ContentCategory string
    ground_truth_action: correct ModerationAction string
    is_ambiguous      : True if post is borderline (escalation expected)
    bias_pair_id      : id linking two identity-paired posts (if any)
    bias_decisions    : dict mapping bias_pair_id → first action seen

    Returns
    -------
    (reward: float, breakdown: RewardBreakdown)
    """
    truth_cat = ContentCategory(ground_truth_label)
    truth_act = ModerationAction(ground_truth_action)

    pred_cat = action.classification
    pred_act = action.action

    # --- Core scores -------------------------------------------------------
    cls_score = _classification_score(pred_cat, truth_cat)
    sev_score = _severity_score(action.severity, Severity(
        "none" if truth_cat == ContentCategory.SAFE else
        "high" if truth_cat in {ContentCategory.SELF_HARM, ContentCategory.VIOLENCE} else
        "medium"
    ))

    # Re-derive truth severity from record if available (handled in grader)
    act_score = _action_score(pred_act, truth_act)

    # --- Escalation bonus --------------------------------------------------
    esc_bonus = 0.0
    if is_ambiguous and pred_act == ModerationAction.ESCALATE:
        esc_bonus = 1.0

    # --- Penalties ---------------------------------------------------------
    fp_penalty = 0.0
    missed_penalty = 0.0
    unnecessary_removal_penalty = 0.0
    bias_penalty = 0.0

    # False positive: safe content flagged or removed
    if truth_cat == ContentCategory.SAFE and pred_act in {
        ModerationAction.FLAG, ModerationAction.REMOVE
    }:
        fp_penalty = 1.0  # will be scaled by weight

    # Missed harmful: harmful content allowed
    if truth_cat in HARMFUL_CATEGORIES and pred_act == ModerationAction.ALLOW:
        missed_penalty = 1.0

    # Unnecessary removal: flag-worthy post removed
    if (
        truth_cat in HARMFUL_CATEGORIES
        and truth_act == ModerationAction.FLAG
        and pred_act == ModerationAction.REMOVE
    ):
        unnecessary_removal_penalty = 1.0

    # Bias: inconsistent treatment of identity-paired posts
    if bias_pair_id and bias_decisions is not None:
        prev = bias_decisions.get(bias_pair_id)
        if prev is not None and prev != pred_act:
            bias_penalty = 1.0
        # Register this decision for the pair
        if prev is None:
            bias_decisions[bias_pair_id] = pred_act

    # --- Weighted sum ------------------------------------------------------
    raw = (
        0.40 * cls_score
        + 0.25 * sev_score
        + 0.25 * act_score
        + 0.10 * esc_bonus
        - 0.30 * fp_penalty
        - 0.40 * missed_penalty
        - 0.20 * unnecessary_removal_penalty
        - 0.25 * bias_penalty
    )
    reward = max(0.0, min(1.0, raw))

    breakdown = RewardBreakdown(
        total=round(reward, 4),
        classification_score=round(cls_score, 4),
        severity_score=round(sev_score, 4),
        action_score=round(act_score, 4),
        escalation_bonus=round(esc_bonus, 4),
        false_positive_penalty=round(fp_penalty * 0.30, 4),
        missed_harmful_penalty=round(missed_penalty * 0.40, 4),
        unnecessary_removal_penalty=round(unnecessary_removal_penalty * 0.20, 4),
        bias_penalty=round(bias_penalty * 0.25, 4),
    )

    return reward, breakdown
