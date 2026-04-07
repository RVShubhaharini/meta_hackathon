# AI Content Moderation OpenEnv

> **A production-grade OpenEnv environment simulating a real-world social media content moderation system — inspired by Meta, YouTube, and X (Twitter) Trust & Safety teams.**

---

## Table of Contents

1. [Project Motivation](#project-motivation)
2. [Environment Overview](#environment-overview)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Task Descriptions](#task-descriptions)
6. [Reward Design](#reward-design)
7. [Advanced Features](#advanced-features)
8. [Project Structure](#project-structure)
9. [Setup & Installation](#setup--installation)
10. [Running Inference](#running-inference)
11. [Docker Usage](#docker-usage)
12. [Hugging Face Spaces](#hugging-face-spaces)
13. [Example Outputs](#example-outputs)

---

## Project Motivation

Content moderation is one of the most critical and challenging problems in the modern internet. Platforms like Meta, YouTube, and Twitter process billions of posts daily. Automated moderation must:

- **Correctly identify** spam, hate speech, harassment, violence, and misinformation
- **Avoid over-moderation** — false positives silence legitimate speech
- **Avoid under-moderation** — missed harmful content damages communities
- **Be unbiased** — consistent decisions regardless of identity groups
- **Support multiple languages** — global platforms serve diverse communities
- **Handle context** — a post's meaning changes within a conversation thread
- **Escalate uncertainty** — borderline cases require human review

This OpenEnv environment allows AI researchers and engineers to benchmark and train content moderation agents against realistic, structured scenarios.

---

## Environment Overview

The environment simulates a moderation queue of social media posts. At each step:

1. The agent receives an **Observation** (post + context + user history)
2. The agent outputs an **Action** (classification + severity + moderation decision)
3. The environment computes a **Reward** based on a deterministic grader
4. The next post is returned, until the episode ends

### OpenEnv Interface

```python
from environment import ModerationEnv

env = ModerationEnv(task_id="task_hard")
obs = env.reset()

while True:
    action = agent.decide(obs)        # Your agent here
    obs, reward, done, info = env.step(action)
    if done:
        break

state = env.state()
print(f"Average reward: {state.cumulative_reward / state.posts_processed:.4f}")
```

---

## Observation Space

```python
class Observation(BaseModel):
    post_id: str                        # Unique identifier
    content: str                        # Post text (en/hi/ta)
    language: Language                  # en | hi | ta | unknown
    user_history: UserHistory           # Past violations, post counts
    user_reputation_score: float        # 0.0 (worst) to 1.0 (best)
    context: ThreadContext              # Parent post, thread history, topic
    metadata: PostMetadata             # Likes, shares, reports, hashtags
    task_id: str                        # Current task
    step_number: int                    # Position in episode
```

**UserHistory** includes:
- `user_id`, `total_posts`, `flagged_count`, `removed_count`
- `prior_violations` — list of past violation categories

**ThreadContext** includes:
- `thread_id`, `parent_post_id`
- `thread_posts` — previous posts in the conversation
- `thread_topic`

---

## Action Space

```python
class Action(BaseModel):
    post_id: str
    classification: ContentCategory    # safe | spam | hate_speech | harassment
                                       # violence | misinformation | sexual | self_harm
    severity: Severity                 # none | low | medium | high
    action: ModerationAction           # allow | flag | remove | escalate
    confidence: float                  # 0.0–1.0
    reasoning: Optional[str]           # Brief explanation
```

### Action Guidelines

| Classification | Severity | Recommended Action |
|---|---|---|
| safe | none | allow |
| spam | low | flag |
| spam | medium/high | remove |
| hate_speech | medium | flag |
| hate_speech | high | remove |
| harassment | medium | flag |
| harassment | high | remove |
| violence | any | remove |
| self_harm | any | escalate |
| ambiguous | any | escalate |

---

## Task Descriptions

### Task Easy — Spam Detection
- **Posts**: 10 | **Languages**: en, hi, ta
- **Categories**: spam, safe
- **Goal**: Binary classification — is this post spam or safe?
- **Scoring**: Correct classification (40%) + correct action (25%) + severity (25%) + escalation (10%)

### Task Medium — Toxicity Classification
- **Posts**: 15 | **Languages**: en, hi, ta  
- **Categories**: safe, spam, hate_speech, harassment, violence, misinformation
- **Goal**: Multi-class toxicity detection with appropriate severity and action
- **Added complexity**: Bias detection pairs (same content, different identity terms)

### Task Hard — Contextual Thread Moderation
- **Posts**: 15 | **Languages**: en, hi, ta
- **Categories**: All categories + self_harm
- **Goal**: Full contextual moderation with thread history, reputation tracking, sarcasm detection, mental health escalation
- **Added complexity**: Thread consistency checks, cross-language posts, ambiguous cases

---

## Reward Design

### Score Components

```
reward = (
    0.40 × classification_score     # Is the category correct?
  + 0.25 × severity_score           # Is the severity level correct?
  + 0.25 × action_score             # Is the moderation action correct?
  + 0.10 × escalation_bonus         # Did agent correctly escalate ambiguous post?
)

# Minus penalties:
  - 0.30  if false positive (safe content flagged/removed)
  - 0.40  if missed harmful content (harmful post allowed through)
  - 0.20  if unnecessary removal (flag-worthy post removed instead)
  - 0.25  if bias violation (inconsistent treatment of identity pairs)

# Clipped to [0.0, 1.0]
```

### Partial Credit

The reward function awards **partial credit** for near-misses:
- Classifying `hate_speech` as `harassment` → 0.5× classification score
- Classifying `harassment` as `harassment` → 1.0× classification score
- Choosing `flag` when `remove` was correct → 0.4× action score

### Bias Detection

The dataset contains **bias pairs**: two posts with identical structure but different identity terms (e.g., ethnicity, religion). The agent must treat both consistently. Inconsistent treatment triggers a 0.25 bias penalty.

---

## Advanced Features

### User Reputation System
- Each user maintains a reputation score [0.0, 1.0]
- Score decays by 0.10 per confirmed violation
- Low-reputation users should receive stricter moderation
- Reputation visible in observation as `user_reputation_score`

### Escalation System
- Ambiguous posts (mental health, borderline content) should be **escalated** rather than allowed or removed
- Correct escalation earns the 0.10 escalation bonus
- Posts marked `is_ambiguous: true` in ground truth expect escalation

### Multi-Language Support
- All three tasks include posts in **English**, **Hindi**, and **Tamil**
- The LLM agent must classify correctly across all languages
- Language code available in observation for routing

### Thread Consistency (Hard Task)
- The hard task penalises contradictory decisions within a thread
- e.g., allowing early posts then suddenly removing a follow-up = consistency penalty
- Penalty injected into grader: 0.10–0.15 per contradiction

---

## Project Structure

```
project/
├── openenv.yaml              # Task metadata, reward spec, compute requirements
├── environment/
│   ├── __init__.py
│   ├── env.py                # ModerationEnv (reset/step/state)
│   ├── models.py             # Pydantic types: Observation, Action, Reward
│   ├── reward.py             # Reward function with all components
│   └── grader.py             # Deterministic per-task graders
├── data/
│   └── dataset.json          # 40-post synthetic multilingual dataset
├── inference.py              # Baseline LLM agent with structured logging
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- An OpenAI-compatible API key (OpenAI, Anthropic via proxy, Together, etc.)

### Install

```bash
git clone https://github.com/example/ai-content-moderation
cd ai-content-moderation
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
export API_BASE_URL="https://api.anthropic.com/v1"   # or https://api.openai.com/v1
export MODEL_NAME="claude-sonnet-4-20250514"          # or gpt-4o, etc.
```

---

## Running Inference

### Run all tasks

```bash
python inference.py --task all
```

### Run a specific task

```bash
python inference.py --task task_easy
python inference.py --task task_medium
python inference.py --task task_hard
```

### Save results to JSON

```bash
python inference.py --task all --output results.json
# Creates: results_task_easy.json, results_task_medium.json, results_task_hard.json
```

### Use the environment directly in Python

```python
from environment import ModerationEnv
from environment.models import Action, ContentCategory, Severity, ModerationAction

env = ModerationEnv(task_id="task_easy")
obs = env.reset()

# Manual agent loop
action = Action(
    post_id=obs.post_id,
    classification=ContentCategory.SPAM,
    severity=Severity.MEDIUM,
    action=ModerationAction.REMOVE,
    confidence=0.9,
    reasoning="Clear spam pattern with urgency triggers and external link"
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward}")
print(f"Breakdown: {info['reward_breakdown']}")
```

---

## Docker Usage

### Build

```bash
docker build -t ai-content-moderation .
```

### Run

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e API_BASE_URL=$API_BASE_URL \
           -e MODEL_NAME=$MODEL_NAME \
           ai-content-moderation
```

### Run a specific task

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e API_BASE_URL=$API_BASE_URL \
           -e MODEL_NAME=$MODEL_NAME \
           ai-content-moderation \
           python inference.py --task task_hard
```

---

## Hugging Face Spaces

To deploy on HF Spaces:

1. Create a new Space (Docker SDK)
2. Push all project files to the Space repository
3. Add your secrets in Settings → Repository secrets:
   - `OPENAI_API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`

The Dockerfile exposes port 7860 as required by HF Spaces.

---

## Example Outputs

### Structured Log Format

```
[START] {"task_id": "task_easy", "model": "claude-sonnet-4-20250514", "timestamp": "...", "total_posts": 10}

[STEP] {"step": 1, "post_id": "easy_001", "action": {"classification": "spam", "severity": "medium", "action": "remove", "confidence": 0.98, "reasoning": "..."}, "reward": 0.9, "ground_truth_label": "spam", "ground_truth_action": "remove", "reward_breakdown": {"total": 0.9, "classification_score": 1.0, "severity_score": 1.0, "action_score": 1.0, ...}}

[STEP] {"step": 2, "post_id": "easy_002", ...}

[END] {"task_id": "task_easy", "average_reward": 0.87, "correct_classifications": 9, "false_positives": 0, "false_negatives": 1, ...}
```

### Expected Scores (Baseline)

| Task | Expected Score | Description |
|---|---|---|
| task_easy | 0.85 | Spam is usually obvious |
| task_medium | 0.72 | Multilingual toxicity harder |
| task_hard | 0.65 | Thread context + bias + escalation |

---

## License

MIT License — see LICENSE for details.
