"""
ModerationEnv — Core OpenEnv environment for AI Content Moderation.

Interface:
    env = ModerationEnv(task_id="task_easy")
    obs = env.reset()
    while True:
        action = agent.decide(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    state = env.state()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .grader import get_grader
from .models import (
    Action,
    EpisodeState,
    Language,
    ModerationAction,
    Observation,
    PostMetadata,
    ThreadContext,
    UserHistory,
)

# Path to the bundled dataset
_DEFAULT_DATASET = Path(__file__).parent.parent / "data" / "dataset.json"

# Sentinel observation returned when the episode is done
_TERMINAL_OBS = Observation(
    post_id="TERMINAL",
    content="",
    language=Language.EN,
    user_history=UserHistory(
        user_id="", total_posts=0, flagged_count=0, removed_count=0
    ),
    user_reputation_score=1.0,
    context=ThreadContext(),
    metadata=PostMetadata(),
    task_id="",
    step_number=-1,
)

# Reputation decay per confirmed violation
_REPUTATION_DECAY = 0.10


def _compute_reputation(record: Dict[str, Any]) -> float:
    """Derive a [0, 1] reputation score from the user history in the record."""
    total = max(record.get("user_total_posts", 1), 1)
    removed = record.get("user_removed_count", 0)
    # Base: 1.0 minus proportional removal rate, decayed by violation count
    violation_count = len(record.get("prior_violations", []))
    base = 1.0 - (removed / total)
    decayed = base - (violation_count * _REPUTATION_DECAY)
    return round(max(0.0, min(1.0, decayed)), 4)


def _record_to_observation(record: Dict[str, Any], task_id: str, step: int) -> Observation:
    """Convert a raw dataset record dict into a typed Observation."""
    user_history = UserHistory(
        user_id=record.get("user_id", "unknown"),
        total_posts=record.get("user_total_posts", 0),
        flagged_count=record.get("user_flagged_count", 0),
        removed_count=record.get("user_removed_count", 0),
        prior_violations=record.get("prior_violations", []),
    )

    context = ThreadContext(
        thread_id=record.get("thread_id"),
        parent_post_id=record.get("parent_post_id"),
        thread_posts=record.get("thread_posts", []),
        thread_topic=record.get("thread_topic"),
    )

    metadata = PostMetadata(
        likes=record.get("likes", 0),
        shares=record.get("shares", 0),
        reports=record.get("reports", 0),
        hashtags=record.get("hashtags", []),
        is_reply=record.get("parent_post_id") is not None,
        timestamp=record.get("timestamp"),
    )

    lang_str = record.get("language", "en")
    try:
        language = Language(lang_str)
    except ValueError:
        language = Language.UNKNOWN

    return Observation(
        post_id=record["post_id"],
        content=record["content"],
        language=language,
        user_history=user_history,
        user_reputation_score=_compute_reputation(record),
        context=context,
        metadata=metadata,
        task_id=task_id,
        step_number=step,
    )


class ModerationEnv:
    """
    OpenEnv-compatible environment for content moderation.

    Parameters
    ----------
    task_id   : "task_easy" | "task_medium" | "task_hard"
    dataset   : path to dataset.json (defaults to bundled data/dataset.json)
    """

    VALID_TASKS = {"task_easy", "task_medium", "task_hard"}

    def __init__(
        self,
        task_id: str = "task_easy",
        dataset: Optional[Path | str] = None,
    ) -> None:
        if task_id not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task_id {task_id!r}. Choose from {sorted(self.VALID_TASKS)}"
            )
        self._task_id = task_id
        self._dataset_path = Path(dataset) if dataset else _DEFAULT_DATASET
        self._all_records: List[Dict[str, Any]] = self._load_dataset()
        self._task_records: List[Dict[str, Any]] = [
            r for r in self._all_records if r.get("task") == task_id
        ]
        if not self._task_records:
            raise RuntimeError(
                f"No records found for task_id={task_id!r} in {self._dataset_path}"
            )

        # Episode state (populated on reset())
        self._grader = get_grader(task_id)
        self._current_index: int = 0
        self._current_step: int = 0
        self._cumulative_reward: float = 0.0
        self._correct_classifications: int = 0
        self._false_positives: int = 0
        self._false_negatives: int = 0
        self._escalations_correct: int = 0
        self._bias_violations: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._dataset_path}. "
                "Ensure data/dataset.json exists."
            )
        with open(self._dataset_path, encoding="utf-8") as f:
            return json.load(f)

    def _reset_stats(self) -> None:
        self._current_index = 0
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._correct_classifications = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._escalations_correct = 0
        self._bias_violations = 0
        self._done = False
        # Fresh grader resets bias/thread tracking
        self._grader = get_grader(self._task_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the first observation."""
        self._reset_stats()
        return _record_to_observation(
            self._task_records[0], self._task_id, step=1
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply the agent's action and return (next_obs, reward, done, info).

        Parameters
        ----------
        action : Action — the agent's moderation decision

        Returns
        -------
        observation : Observation (next post or TERMINAL)
        reward      : float in [0.0, 1.0]
        done        : bool — True when all posts have been processed
        info        : dict with ground_truth, reward_breakdown, etc.
        """
        if self._done:
            return _TERMINAL_OBS, 0.0, True, {"error": "Episode already done. Call reset()."}

        record = self._task_records[self._current_index]

        # Grade the action
        reward, breakdown = self._grader.grade(action, record)
        self._cumulative_reward += reward
        self._current_step += 1

        # Update episode statistics
        truth_cat = record["label"]
        truth_act = record["expected_action"]

        if action.classification.value == truth_cat:
            self._correct_classifications += 1

        # False positive: safe content actioned
        if truth_cat == "safe" and action.action in {
            ModerationAction.FLAG, ModerationAction.REMOVE
        }:
            self._false_positives += 1

        # False negative: harmful content allowed
        if truth_cat != "safe" and action.action == ModerationAction.ALLOW:
            self._false_negatives += 1

        # Correct escalation on ambiguous posts
        if record.get("is_ambiguous") and action.action == ModerationAction.ESCALATE:
            self._escalations_correct += 1

        # Bias violation tracking (grader populates bias_decisions)
        if breakdown.get("bias_penalty", 0) > 0:
            self._bias_violations += 1

        info = {
            "ground_truth_label": truth_cat,
            "ground_truth_action": truth_act,
            "is_ambiguous": record.get("is_ambiguous", False),
            "bias_pair_id": record.get("bias_pair_id"),
            "reward_breakdown": breakdown,
        }

        # Advance to next record
        self._current_index += 1
        if self._current_index >= len(self._task_records):
            self._done = True
            return _TERMINAL_OBS, reward, True, info

        next_obs = _record_to_observation(
            self._task_records[self._current_index],
            self._task_id,
            step=self._current_step + 1,
        )
        return next_obs, reward, False, info

    def state(self) -> EpisodeState:
        """Return the current episode statistics."""
        return EpisodeState(
            task_id=self._task_id,
            current_step=self._current_step,
            posts_processed=self._current_index,
            cumulative_reward=round(self._cumulative_reward, 4),
            correct_classifications=self._correct_classifications,
            false_positives=self._false_positives,
            false_negatives=self._false_negatives,
            escalations_correct=self._escalations_correct,
            bias_violations=self._bias_violations,
            done=self._done,
        )
