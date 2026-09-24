"""Tests for analytics requirements."""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.settings import ActionSteps, RequirementBounds  # noqa: E402
from argos.domain.requirements import AnalyticsRequirements, RequirementAction, bucketize  # noqa: E402


class TestBucketize(unittest.TestCase):
    """Tests for bucketize function."""

    def test_bucketize_below_first_edge(self):
        """Value below first edge should return 0."""
        edges = (25, 50, 75)
        self.assertEqual(bucketize(10.0, edges), 0)

    def test_bucketize_at_edge(self):
        """Value at edge should return that bucket."""
        edges = (25, 50, 75)
        self.assertEqual(bucketize(25.0, edges), 0)

    def test_bucketize_between_edges(self):
        """Value between edges should return correct bucket."""
        edges = (25, 50, 75)
        self.assertEqual(bucketize(30.0, edges), 1)
        self.assertEqual(bucketize(60.0, edges), 2)

    def test_bucketize_above_all_edges(self):
        """Value above all edges should return last bucket + 1."""
        edges = (25, 50, 75)
        self.assertEqual(bucketize(100.0, edges), 3)

    def test_bucketize_rejects_non_finite_values_and_unsorted_edges(self):
        with self.assertRaises(ValueError):
            bucketize(math.nan, (25, 50, 75))
        with self.assertRaises(ValueError):
            bucketize(30.0, (50, 25, 75))


class TestAnalyticsRequirements(unittest.TestCase):
    """Tests for AnalyticsRequirements."""

    def setUp(self):
        """Set up bounds and steps."""
        self.bounds = RequirementBounds(
            coverage=(0.2, 1.0),
            sample=(0.1, 1.0),
            freshness_seconds=(5.0, 600.0),
        )
        self.steps = ActionSteps(coverage=0.05, sample=0.05, freshness_seconds=5.0)

    def test_default_values(self):
        """Default requirements should have expected values."""
        req = AnalyticsRequirements()
        self.assertEqual(req.coverage, 0.6)
        self.assertEqual(req.sample, 0.5)
        self.assertEqual(req.freshness_seconds, 60.0)

    def test_clamp_enforces_bounds(self):
        """clamp() should enforce min/max bounds."""
        req = AnalyticsRequirements(coverage=1.5, sample=-0.1, freshness_seconds=1.0)
        req.clamp(self.bounds)

        self.assertEqual(req.coverage, 1.0)
        self.assertEqual(req.sample, 0.1)
        self.assertEqual(req.freshness_seconds, 5.0)

    def test_apply_action_increase_coverage(self):
        """INCREASE_COVERAGE should add step to coverage."""
        req = AnalyticsRequirements(coverage=0.5)
        req.apply_action(RequirementAction.INCREASE_COVERAGE, self.bounds, self.steps)
        self.assertAlmostEqual(req.coverage, 0.55, places=5)

    def test_apply_action_decrease_coverage(self):
        """DECREASE_COVERAGE should subtract step from coverage."""
        req = AnalyticsRequirements(coverage=0.5)
        req.apply_action(RequirementAction.DECREASE_COVERAGE, self.bounds, self.steps)
        self.assertAlmostEqual(req.coverage, 0.45, places=5)

    def test_apply_action_increase_sample(self):
        """INCREASE_SAMPLE should add step to sample."""
        req = AnalyticsRequirements(sample=0.5)
        req.apply_action(RequirementAction.INCREASE_SAMPLE, self.bounds, self.steps)
        self.assertAlmostEqual(req.sample, 0.55, places=5)

    def test_apply_action_decrease_sample(self):
        """DECREASE_SAMPLE should subtract step from sample."""
        req = AnalyticsRequirements(sample=0.5)
        req.apply_action(RequirementAction.DECREASE_SAMPLE, self.bounds, self.steps)
        self.assertAlmostEqual(req.sample, 0.45, places=5)

    def test_apply_action_make_fresher(self):
        """MAKE_FRESHER should reduce freshness_seconds."""
        req = AnalyticsRequirements(freshness_seconds=60.0)
        req.apply_action(RequirementAction.MAKE_FRESHER, self.bounds, self.steps)
        self.assertAlmostEqual(req.freshness_seconds, 55.0, places=5)

    def test_apply_action_make_staler(self):
        """MAKE_STALER should increase freshness_seconds."""
        req = AnalyticsRequirements(freshness_seconds=60.0)
        req.apply_action(RequirementAction.MAKE_STALER, self.bounds, self.steps)
        self.assertAlmostEqual(req.freshness_seconds, 65.0, places=5)

    def test_apply_action_hold(self):
        """HOLD should not change requirements."""
        req = AnalyticsRequirements(coverage=0.5, sample=0.5, freshness_seconds=60.0)
        req.apply_action(RequirementAction.HOLD, self.bounds, self.steps)
        self.assertAlmostEqual(req.coverage, 0.5, places=5)
        self.assertAlmostEqual(req.sample, 0.5, places=5)
        self.assertAlmostEqual(req.freshness_seconds, 60.0, places=5)

    def test_as_vector(self):
        """as_vector should return dict with numeric values."""
        req = AnalyticsRequirements(coverage=0.6, sample=0.5, freshness_seconds=60.0)
        vec = req.as_vector()
        self.assertEqual(vec["coverage"], 0.6)
        self.assertEqual(vec["sample"], 0.5)
        self.assertEqual(vec["freshness_seconds"], 60.0)

    def test_as_state_tuple(self):
        """as_state_tuple should return tuple of bucket indices."""
        req = AnalyticsRequirements(coverage=0.6, sample=0.5, freshness_seconds=60.0)
        state = req.as_state_tuple()
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 3)
        self.assertTrue(all(isinstance(x, int) for x in state))

    def test_freshness_pressure(self):
        """freshness_pressure should be inverse of freshness_seconds."""
        req = AnalyticsRequirements(freshness_seconds=100.0)
        self.assertAlmostEqual(req.freshness_pressure(), 0.01, places=5)

        req2 = AnalyticsRequirements(freshness_seconds=1.0)
        self.assertAlmostEqual(req2.freshness_pressure(), 1.0, places=5)

    def test_estimated_cost(self):
        """estimated_cost should return a float based on requirements."""
        req = AnalyticsRequirements(coverage=0.5, sample=0.5, freshness_seconds=100.0)
        cost = req.estimated_cost()
        # cost = 0.5 * 0.5 * (1 + 0.01) = 0.2525
        self.assertAlmostEqual(cost, 0.2525, places=4)

    def test_requirement_quality(self):
        """requirement_quality should return normalized score."""
        req = AnalyticsRequirements(coverage=1.0, sample=1.0, freshness_seconds=5.0)
        quality = req.requirement_quality(self.bounds)
        # Maximum quality
        self.assertGreater(quality, 0.0)
        self.assertLessEqual(quality, 1.0)


if __name__ == "__main__":
    unittest.main()
