"""
Deterministic graders for each task in AI Content Moderation OpenEnv.

Each grader wraps compute_reward with task-specific logic and
state tracking (thread consistency, bias pair management).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    ContentCategory,
    ModerationAction,
    Severity,
)
from .reward import compute_reward


class BaseGrader:
    """Shared grading infrastructure."""

    def __init__(self) -> None:
        self.bias_decisions: Dict[str, ModerationAction] = {}
        self.thread_decisions: Dict[str, List[Tuple[str, ModerationAction]]] = {}
        self.lang_decisions: Dict[str, ModerationAction] = {}

    def _check_thread_consistency(
        self,
        thread_id: Optional[str],
        post_id: str,
        action: Action,
    ) -> float:
        """
        Returns a penalty (0.0–0.15) if this decision contradicts
        an earlier decision in the same thread.
        """
        if not thread_id:
            return 0.0

        history = self.thread_decisions.setdefault(thread_id, [])
        penalty = 0.0

        for prev_post_id, prev_action in history:
            # If we allowed a post but now remove a near-identical safe follow-up → inconsistent
            # If we removed a post but now allow the same type → inconsistent
            if (
                prev_action == ModerationAction.REMOVE
                and action.action == ModerationAction.ALLOW
                and action.classification
                in {ContentCategory.SAFE}
            ):
                pass  # Allow → Remove escalation is fine
            elif (
                prev_action == ModerationAction.ALLOW
                and action.action == ModerationAction.REMOVE
                and action.classification == ContentCategory.SAFE
            ):
                penalty = 0.10  # Removing safe content after allowing similar = inconsistent

        history.append((post_id, action.action))
        return penalty

    def grade(
        self,
        action: Action,
        record: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        raise NotImplementedError


class EasyGrader(BaseGrader):
    """
    task_easy — Spam Detection.
    Binary: spam vs safe. No bias pairs, no thread context.
    """

    def grade(
        self,
        action: Action,
        record: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label=record["label"],
            ground_truth_action=record["expected_action"],
            is_ambiguous=record.get("is_ambiguous", False),
            bias_pair_id=None,
            bias_decisions=None,
            user_reputation=record.get("user_reputation", 1.0),
        )
        return reward, breakdown.model_dump()


class MediumGrader(BaseGrader):
    """
    task_medium — Toxicity Classification.
    Multi-class with bias detection pairs.
    """

    def grade(
        self,
        action: Action,
        record: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label=record["label"],
            ground_truth_action=record["expected_action"],
            is_ambiguous=record.get("is_ambiguous", False),
            bias_pair_id=record.get("bias_pair_id"),
            bias_decisions=self.bias_decisions,
            user_reputation=record.get("user_reputation", 1.0),
            lang_pair_id=record.get("lang_pair_id"),
            lang_decisions=self.lang_decisions,
        )
        return reward, breakdown.model_dump()


class HardGrader(BaseGrader):
    """
    task_hard — Contextual Thread Moderation.
    Full feature set: bias detection, escalation system, thread consistency.
    """

    def grade(
        self,
        action: Action,
        record: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        thread_penalty = self._check_thread_consistency(
            thread_id=record.get("thread_id"),
            post_id=record["post_id"],
            action=action,
        )

        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label=record["label"],
            ground_truth_action=record["expected_action"],
            is_ambiguous=record.get("is_ambiguous", False),
            bias_pair_id=record.get("bias_pair_id"),
            bias_decisions=self.bias_decisions,
            user_reputation=record.get("user_reputation", 1.0),
            lang_pair_id=record.get("lang_pair_id"),
            lang_decisions=self.lang_decisions,
        )

        # Apply thread consistency penalty
        if thread_penalty > 0:
            reward = max(0.0, reward - thread_penalty)
            breakdown_dict = breakdown.model_dump()
            breakdown_dict["thread_consistency_penalty"] = round(thread_penalty, 4)
            breakdown_dict["total"] = round(reward, 4)
        else:
            breakdown_dict = breakdown.model_dump()
            breakdown_dict["thread_consistency_penalty"] = 0.0

        return reward, breakdown_dict


GRADERS = {
    "task_easy": EasyGrader,
    "task_medium": MediumGrader,
    "task_hard": HardGrader,
}


def get_grader(task_id: str) -> BaseGrader:
    cls = GRADERS.get(task_id)
    if cls is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(GRADERS)}")
    return cls()
