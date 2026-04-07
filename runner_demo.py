from environment.env import ModerationEnv
from environment.models import Action, ContentCategory, Severity, ModerationAction
import json

print("\n🚀 Starting AI Content Moderation OpenEnv Demo")
print("-" * 50)

# Initialize the environment with 'task_easy'
env = ModerationEnv(task_id="task_easy")
obs = env.reset()

print("📦 Observation Received:")
print(f"Post ID: {obs.post_id}")
print(f"Content: '{obs.content}'")
print(f"User History: Flagged {obs.user_history.flagged_count} times")
print("-" * 50)

print("🤖 Agent processing observation... (Simulated Action)")
# Simulate the AI making a decision
action = Action(
    post_id=obs.post_id,
    classification=ContentCategory.SPAM,
    severity=Severity.HIGH,
    action=ModerationAction.REMOVE,
    confidence=0.95,
    reasoning="Post contains spam keywords ('FREE iPhone giveaway', 'Click this link immediately') and user has a history of spam violations."
)

# Step the environment
obs, reward, done, info = env.step(action)

print("-" * 50)
print("✅ Action Computed & Graded:")
print(f"Reward Score: {reward:.2f}/1.0")
print("Detailed Reward Breakdown:")
print(json.dumps(info['reward_breakdown'], indent=2))
print("=" * 50)
