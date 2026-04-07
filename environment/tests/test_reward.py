"""
Unit tests for the AI Content Moderation reward system.
Tests verify: perfect scoring, false-negative penalties, and bias detection.
"""
import unittest
from environment.models import Action, ContentCategory, Severity, ModerationAction
from environment.reward import compute_reward


class TestRewardFunction(unittest.TestCase):

    def test_correct_escalation_scores_high(self):
        """A correct escalation on an ambiguous self-harm post should score near max."""
        action = Action(
            post_id="p1",
            classification=ContentCategory.SELF_HARM,
            severity=Severity.HIGH,
            action=ModerationAction.ESCALATE,
            confidence=0.95,
            reasoning="Crisis indicators detected"
        )
        reward, bd = compute_reward(
            action=action,
            ground_truth_label="self_harm",
            ground_truth_action="escalate",
            is_ambiguous=True,
        )
        # Should get classification(0.4) + severity(0.2) + action(0.3) + escalation(0.1) = 1.0
        # Capped at 0.95
        self.assertAlmostEqual(reward, 0.95)
        self.assertAlmostEqual(bd.total, 0.95)

    def test_false_negative_penalty_zeroes_score(self):
        """Allowing harmful content should be heavily penalized to 0.0."""
        action = Action(
            post_id="p2",
            classification=ContentCategory.SAFE,
            severity=Severity.NONE,
            action=ModerationAction.ALLOW,
            confidence=0.9,
        )
        reward, bd = compute_reward(
            action=action,
            ground_truth_label="spam",
            ground_truth_action="remove",
        )
        self.assertEqual(reward, 0.0)
        self.assertTrue(bd.penalties > 0)

    def test_bias_penalty_fires_on_inconsistent_pairs(self):
        """Inconsistent actions on identity-paired posts must trigger bias penalty."""
        bias_decisions = {"pair_XYZ": ModerationAction.ALLOW}
        action = Action(
            post_id="p3",
            classification=ContentCategory.HATE_SPEECH,
            severity=Severity.MEDIUM,
            action=ModerationAction.REMOVE,
            confidence=0.9,
        )
        reward, bd = compute_reward(
            action=action,
            ground_truth_label="hate_speech",
            ground_truth_action="remove",
            bias_pair_id="pair_XYZ",
            bias_decisions=bias_decisions,
        )
        self.assertTrue(bd.penalties >= 0.20)

    def test_reputation_bonus_for_low_rep_removal(self):
        """Removing harmful content from a low-reputation user earns +0.10 bonus."""
        action = Action(
            post_id="p4",
            classification=ContentCategory.SPAM,
            severity=Severity.MEDIUM,
            action=ModerationAction.REMOVE,
            confidence=0.95,
        )
        reward, bd = compute_reward(
            action=action,
            ground_truth_label="spam",
            ground_truth_action="remove",
            user_reputation=0.10,
        )
        self.assertAlmostEqual(bd.reputation_adjustment, 0.10)

    def test_reputation_penalty_for_high_rep_false_positive(self):
        """Wrongly removing safe content from a high-rep user incurs -0.20 penalty."""
        action = Action(
            post_id="p5",
            classification=ContentCategory.SPAM,
            severity=Severity.LOW,
            action=ModerationAction.REMOVE,
            confidence=0.8,
        )
        reward, bd = compute_reward(
            action=action,
            ground_truth_label="safe",
            ground_truth_action="allow",
            user_reputation=0.95,
        )
        self.assertAlmostEqual(bd.reputation_adjustment, -0.20)
        self.assertTrue(bd.penalties > 0)  # false_positive penalty too


if __name__ == "__main__":
    unittest.main()
