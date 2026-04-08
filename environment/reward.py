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
    user_reputation: float = 1.0,
    lang_pair_id: str | None = None,
    lang_decisions: dict | None = None,
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
    cross_lingual_penalty = 0.0

    # Removed conf_multiplier since Action schema is strict

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
            
    # Cross-lingual consistency
    if lang_pair_id and lang_decisions is not None:
        prev_lang = lang_decisions.get(lang_pair_id)
        if prev_lang is not None and prev_lang != pred_act:
            cross_lingual_penalty = 1.0
        if prev_lang is None:
            lang_decisions[lang_pair_id] = pred_act

    # --- Reputation Modifiers ----------------------------------------------
    reputation_bonus = 0.0
    if user_reputation < 0.3 and truth_cat in HARMFUL_CATEGORIES and pred_act == ModerationAction.REMOVE:
        reputation_bonus = 0.10

    reputation_penalty = 0.0
    if user_reputation > 0.8 and truth_cat == ContentCategory.SAFE and pred_act == ModerationAction.REMOVE:
        reputation_penalty = 0.20

    reputation_adjustment = reputation_bonus - reputation_penalty

    # --- Weighted sum ------------------------------------------------------
    c_score = 0.40 * cls_score
    s_score = 0.20 * sev_score
    a_score = 0.30 * act_score
    
    # Dynamic reward: increase escalation bonus for ambiguous cases
    dynamic_e = 1.5 if (is_ambiguous and esc_bonus > 0.0 and truth_cat in HARMFUL_CATEGORIES) else 1.0
    e_score = 0.10 * esc_bonus * dynamic_e
    
    pen = (fp_penalty * 0.30) + (missed_penalty * 0.40) + (unnecessary_removal_penalty * 0.20) + (bias_penalty * 0.20)

    base_score = c_score + s_score + a_score + e_score
    raw = base_score - pen - (cross_lingual_penalty * 0.25) + reputation_adjustment
    # Strict OpenEnv constraint: score must be in (0, 1)
    reward = max(0.0001, min(0.9999, raw))

    breakdown = RewardBreakdown(
        total=round(reward, 4),
        classification_score=round(c_score, 4),
        severity_score=round(s_score, 4),
        action_score=round(a_score, 4),
        escalation_score=round(e_score, 4),
        reputation_adjustment=round(reputation_adjustment, 4),
        cross_lingual_penalty=round((cross_lingual_penalty * 0.25), 4),
        penalties=round(pen, 4),
    )

    return reward, breakdown
