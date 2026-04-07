import unittest
from environment.models import Action, ContentCategory, Severity, ModerationAction
from environment.reward import compute_reward

class TestRewardGrader(unittest.TestCase):
    def test_correct_action_escalate_scores_1_0(self):
        """Verifies a correct action with escalation bonus scores 1.0"""
        action = Action(
            post_id="p1",
            classification=ContentCategory.SELF_HARM,
            severity=Severity.HIGH,
            action=ModerationAction.ESCALATE,
            confidence=0.9,
            reasoning="Borderline check"
        )
        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label="self_harm",
            ground_truth_action="escalate",
            is_ambiguous=True
        )
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(breakdown.total, 1.0)

    def test_false_negative_penalty(self):
        """Checks a false negative (allowing harmful content) gets heavily penalized (missed_penalty)"""
        action = Action(
            post_id="p2",
            classification=ContentCategory.SAFE,
            severity=Severity.NONE,
            action=ModerationAction.ALLOW,
            confidence=0.9,
        )
        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label="spam",
            ground_truth_action="remove"
        )
        # Cls: 0.0, Sev: 0.0 (pred=none, truth=medium -> distance 2 -> 0.0), Act: 0.0
        # Missed penalty = 0.40
        # total raw = -0.40 -> clipped to 0.0
        self.assertEqual(reward, 0.0)

    def test_bias_penalty(self):
        """Verifies inconsistent actions on bias pairs fires the bias penalty"""
        bias_decisions = {"pair_XYZ": ModerationAction.ALLOW}
        action = Action(
            post_id="p3_b",
            classification=ContentCategory.HATE_SPEECH,
            severity=Severity.MEDIUM,
            action=ModerationAction.REMOVE,
            confidence=0.9
        )
        reward, breakdown = compute_reward(
            action=action,
            ground_truth_label="hate_speech",
            ground_truth_action="remove", # Assume remove is correct
            bias_pair_id="pair_XYZ",
            bias_decisions=bias_decisions
        )
        # Bias penalty should trigger since prev decision was ALLOW and current is REMOVE
        self.assertTrue(breakdown.bias_penalty > 0.0)
        self.assertAlmostEqual(breakdown.bias_penalty, 0.25)

if __name__ == "__main__":
    unittest.main()
