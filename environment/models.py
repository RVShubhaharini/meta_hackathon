"""
Typed Pydantic models for AI Content Moderation OpenEnv.
Defines Observation, Action, Reward, and all supporting types.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Language(str, Enum):
    EN = "en"
    HI = "hi"
    TA = "ta"
    UNKNOWN = "unknown"


class ContentCategory(str, Enum):
    SAFE = "safe"
    SPAM = "spam"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    MISINFORMATION = "misinformation"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModerationAction(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    REMOVE = "remove"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Observation sub-models
# ---------------------------------------------------------------------------

class UserHistory(BaseModel):
    user_id: str
    total_posts: int
    flagged_count: int
    removed_count: int
    prior_violations: List[str] = Field(default_factory=list)


class ThreadContext(BaseModel):
    thread_id: Optional[str] = None
    parent_post_id: Optional[str] = None
    thread_posts: List[str] = Field(default_factory=list)
    thread_topic: Optional[str] = None


class PostMetadata(BaseModel):
    likes: int = 0
    shares: int = 0
    reports: int = 0
    hashtags: List[str] = Field(default_factory=list)
    is_reply: bool = False
    timestamp: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation  (state given to the agent each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    post_id: str
    content: str
    language: Language
    user_history: UserHistory
    user_reputation_score: float = Field(ge=0.0, le=1.0)
    context: ThreadContext
    metadata: PostMetadata
    task_id: str
    step_number: int


# ---------------------------------------------------------------------------
# Action  (agent output each step)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    post_id: str
    classification: ContentCategory
    severity: Severity
    action: ModerationAction


# ---------------------------------------------------------------------------
# Reward breakdown  (returned in info dict each step)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    total: float
    classification_score: float
    severity_score: float
    action_score: float
    escalation_score: float
    reputation_adjustment: float
    cross_lingual_penalty: float
    penalties: float


# ---------------------------------------------------------------------------
# Episode state  (returned by env.state())
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_id: str
    current_step: int
    total_posts_processed: int
    cumulative_reward: float
    correct_moderations: int
    false_positives: int
    missed_harmful_content: int
    escalation_cases: int
    bias_violations: int
    cross_lingual_violations: int
    precision: float
    recall: float
    f1_score: float
    escalation_accuracy: float
    done: bool
