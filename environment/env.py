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
        user_reputation_score=record.get('user_reputation', 1.0),
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
        self._correct_moderations: int = 0
        self._false_positives: int = 0
        self._missed_harmful_content: int = 0
        self._escalation_cases: int = 0
        self._bias_violations: int = 0
        self._cross_lingual_violations: int = 0
        self._true_positives: int = 0
        self._correct_escalations: int = 0
        self._expected_escalations: int = 0
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
        self._correct_moderations = 0
        self._false_positives = 0
        self._missed_harmful_content = 0
        self._escalation_cases = 0
        self._bias_violations = 0
        self._cross_lingual_violations = 0
        self._true_positives = 0
        self._correct_escalations = 0
        self._expected_escalations = 0
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
            self._correct_moderations += 1

        # Precision/Recall trackers
        is_harmful = truth_cat != "safe"
        agent_actioned = action.action in {ModerationAction.FLAG, ModerationAction.REMOVE, ModerationAction.ESCALATE}
        
        if is_harmful and agent_actioned:
            self._true_positives += 1
        
        if not is_harmful and agent_actioned:
            self._false_positives += 1
            
        if is_harmful and not agent_actioned:
            self._missed_harmful_content += 1

        # Escalation trackers
        if record.get("is_ambiguous"):
            self._expected_escalations += 1
            if action.action == ModerationAction.ESCALATE:
                self._correct_escalations += 1
                self._escalation_cases += 1

        # Bias violation tracking
        if record.get("bias_pair_id") and breakdown.get("penalties", 0) > 0:
            self._bias_violations += 1
            
        # Cross-lingual violation tracking
        if breakdown.get("cross_lingual_penalty", 0) > 0:
            self._cross_lingual_violations += 1

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
        precision = 0.0
        if (self._true_positives + self._false_positives) > 0:
            precision = self._true_positives / (self._true_positives + self._false_positives)
            
        recall = 0.0
        if (self._true_positives + self._missed_harmful_content) > 0:
            recall = self._true_positives / (self._true_positives + self._missed_harmful_content)
            
        f1_score = 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        esc_acc = 0.0
        if self._expected_escalations > 0:
            esc_acc = self._correct_escalations / self._expected_escalations
            
        return EpisodeState(
            task_id=self._task_id,
            current_step=self._current_step,
            total_posts_processed=self._current_index,
            cumulative_reward=round(self._cumulative_reward, 4),
            correct_moderations=self._correct_moderations,
            false_positives=self._false_positives,
            missed_harmful_content=self._missed_harmful_content,
            escalation_cases=self._escalation_cases,
            bias_violations=self._bias_violations,
            cross_lingual_violations=self._cross_lingual_violations,
            precision=round(max(0.0001, min(0.9999, precision)), 4),
            recall=round(max(0.0001, min(0.9999, recall)), 4),
            f1_score=round(max(0.0001, min(0.9999, f1_score)), 4),
            escalation_accuracy=round(max(0.0001, min(0.9999, esc_acc)), 4),
            done=self._done,
        )
