"""Tests for the RL agent registry and backends."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.domain.requests import AnalyticsRequest  # noqa: E402
from argos.orchestrator.loop import OrchestrationLoop, OrchestrationLoopConfig  # noqa: E402
from argos.orchestrator.rl.factory import create_agent, is_torch_available  # noqa: E402
from argos.orchestrator.rl.qlearning import QLearningConfig, TabularQLearningAgent  # noqa: E402


class TestTabularQLearningAgent(unittest.TestCase):
    """Tests for TabularQLearningAgent."""

    def setUp(self):
        self.config = QLearningConfig(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.3,
            exploration_decay=0.9,
            min_exploration_rate=0.01,
            seed=42,
        )
        self.agent = TabularQLearningAgent(self.config)

    def test_select_action_greedy(self):
        """With epsilon=0, agent should always select best action."""
        self.agent.set_epsilon(0.0)
        state = (1, 1, 1)
        action = self.agent.select_action(state, training=True)
        self.assertFalse(self.agent.last_was_exploration)
        self.assertIsNotNone(action)

    def test_select_action_with_exploration(self):
        """With epsilon=1, agent should explore randomly."""
        self.agent.set_epsilon(1.0)
        state = (1, 1, 1)
        self.agent.select_action(state, training=True)
        self.assertTrue(self.agent.last_was_exploration)

    def test_select_action_no_training(self):
        """With training=False, exploration is disabled."""
        self.agent.set_epsilon(1.0)
        state = (1, 2, 3)
        self.agent.select_action(state, training=False)
        self.assertFalse(self.agent.last_was_exploration)

    def test_update_changes_q_value(self):
        """Q-table update should modify values according to Bellman equation."""
        from argos.orchestrator.rl.base import STANDARD_ACTIONS

        state = (1, 1, 1)
        next_state = (1, 1, 2)
        action = STANDARD_ACTIONS[1]  # increase_coverage
        reward = 1.0

        initial_q = self.agent.get_q_value(state, action)
        self.assertEqual(initial_q, 3.0)

        self.agent.update(state, action, reward, next_state)

        new_q = self.agent.get_q_value(state, action)
        # Q(s,a) = (1-0.1)*3 + 0.1*(1.0 + 0.9*3) = 3.07
        self.assertAlmostEqual(new_q, 3.07, places=5)

    def test_table_snapshot_returns_copy(self):
        """table_snapshot should return current Q-table."""
        state = (1, 1, 1)
        self.agent.select_action(state)
        snapshot = self.agent.table_snapshot()
        self.assertIn(state, snapshot)

    def test_set_epsilon_syncs_base_class(self):
        """set_epsilon must keep base class _exploration_rate in sync."""
        self.agent.set_epsilon(0.42)
        self.assertAlmostEqual(self.agent.epsilon, 0.42)
        self.assertEqual(self.agent.epsilon, self.agent._exploration_rate)

    def test_decay_exploration_syncs_base_class(self):
        """decay_exploration must keep base class _exploration_rate in sync."""
        initial = self.agent.epsilon
        self.agent.decay_exploration()
        self.assertLess(self.agent.epsilon, initial)
        self.assertEqual(self.agent.epsilon, self.agent._exploration_rate)

    def test_get_metrics_has_consistent_epsilon(self):
        """get_metrics() exploration_rate must match agent.epsilon."""
        self.agent.set_epsilon(0.25)
        metrics = self.agent.get_metrics()
        self.assertAlmostEqual(metrics["epsilon"], 0.25)
        self.assertAlmostEqual(metrics["exploration_rate"], 0.25)

    def test_state_count(self):
        """state_count tracks unique states in Q-table."""
        self.agent.set_epsilon(0.0)  # force exploitation so Q-table is accessed
        self.assertEqual(self.agent.state_count, 0)
        self.agent.select_action((1, 2, 3))
        self.assertEqual(self.agent.state_count, 1)
        self.agent.select_action((4, 5, 6))
        self.assertEqual(self.agent.state_count, 2)

    def test_get_best_action(self):
        """get_best_action returns the greedy action."""
        from argos.orchestrator.rl.base import STANDARD_ACTIONS

        state = (1, 2, 3)
        action = STANDARD_ACTIONS[3]  # increase_sample
        self.agent.update(state, action, reward=10.0, next_state=state)

        best = self.agent.get_best_action(state)
        self.assertEqual(best, action)


class TestRLAgentFactory:
    """Tests for runtime algorithm selection and persistence."""

    def test_factory_creates_qlearning_agent(self):
        """Q-learning should stay available without optional ML extras."""
        agent, metadata = create_agent(
            algorithm="qlearning",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.2,
            exploration_decay=0.95,
            min_exploration_rate=0.01,
            seed=7,
        )

        assert isinstance(agent, TabularQLearningAgent)
        assert metadata["requested_algorithm"] == "qlearning"
        assert metadata["effective_algorithm"] == "qlearning"
        assert metadata["fallback_reason"] is None

    def test_factory_falls_back_to_qlearning_when_optional_backend_unavailable(self, monkeypatch):
        """Missing torch backend should degrade to qlearning, not crash the runtime."""
        monkeypatch.setattr("argos.orchestrator.rl.factory.is_torch_available", lambda: False)

        agent, metadata = create_agent(
            algorithm="dqn",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.2,
            exploration_decay=0.95,
            min_exploration_rate=0.01,
            seed=9,
        )

        assert isinstance(agent, TabularQLearningAgent)
        assert metadata["requested_algorithm"] == "dqn"
        assert metadata["effective_algorithm"] == "qlearning"
        assert "torch" in metadata["fallback_reason"].lower()

    @pytest.mark.skipif(not is_torch_available(), reason="torch optional backend not installed")
    @pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
    def test_factory_creates_torch_agents(self, algorithm):
        """DQN and PPO should be constructible when torch is available."""
        agent, metadata = create_agent(
            algorithm=algorithm,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=0.15,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
            seed=13,
        )

        metrics = agent.get_metrics()
        assert metadata["effective_algorithm"] == algorithm
        assert metrics["algorithm"] == algorithm
        assert metrics["policy_version"].startswith(f"{algorithm}-v")
        assert metrics["state_schema_version"] == "argos.mdp.v4"
        assert metrics["action_schema_version"] == "argos.actions.masked-discrete.v2"

    def test_loop_initializes_requested_algorithm_for_request(self):
        """The orchestration loop should provision the requested agent backend."""
        if not is_torch_available():
            pytest.skip("torch optional backend not installed")

        loop = OrchestrationLoop(
            config=OrchestrationLoopConfig(
                enable_logging=False,
                enable_persistence=False,
                enable_control_plane=False,
                enable_rl=True,
            )
        )
        loop.register_node("node-1", "http://node-1")
        request = AnalyticsRequest(request_id="req-dqn", algorithm="dqn")
        loop.submit_request(request)

        metrics = loop.get_rl_metrics()
        assert metrics["req-dqn"]["algorithm"] == "dqn"
        assert metrics["req-dqn"]["requested_algorithm"] == "dqn"
        assert metrics["req-dqn"]["fallback_reason"] is None

    def test_qlearning_checkpoint_roundtrip(self):
        """Q-learning policies should survive save/load roundtrip."""
        from argos.orchestrator.rl.base import STANDARD_ACTIONS

        agent, _ = create_agent(
            algorithm="qlearning",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.0,
            exploration_decay=0.95,
            min_exploration_rate=0.01,
            seed=21,
        )
        state = (1, 2, 3)
        agent.update(state, STANDARD_ACTIONS[1], 1.0, state)
        before = agent.get_policy(state)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            agent.save(str(path))
            restored, _ = create_agent(
                algorithm="qlearning",
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=0.0,
                exploration_decay=0.95,
                min_exploration_rate=0.01,
                seed=21,
            )
            restored.load(str(path))
            after = restored.get_policy(state)

        assert {a.index: v for a, v in before.items()} == {a.index: v for a, v in after.items()}

    def test_qlearning_rejects_legacy_state_schema(self):
        """Policies trained against the 11-dimensional state must not load."""
        import json

        agent, _ = create_agent(
            algorithm="qlearning",
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.0,
            exploration_decay=0.95,
            min_exploration_rate=0.01,
            seed=21,
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            agent.save(str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["policy_metadata"]["state_schema_version"] = "argos.mdp.v1"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with pytest.raises(ValueError, match="Incompatible state schema"):
                agent.load(str(path))

    @pytest.mark.skipif(not is_torch_available(), reason="torch optional backend not installed")
    @pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
    def test_torch_checkpoint_roundtrip(self, algorithm):
        """Torch-backed policies should survive save/load roundtrip."""
        state = (0, 1, 2, 3, 4, 1)

        agent, _ = create_agent(
            algorithm=algorithm,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=0.1,
            exploration_decay=0.99,
            min_exploration_rate=0.01,
            seed=33,
        )
        action = agent.select_action(state, training=False)
        agent.update(state, action, 0.75, state, done=False)
        before = {a.index: v for a, v in agent.get_policy(state).items()}

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"policy_{algorithm}.pt"
            agent.save(str(path))
            restored, _ = create_agent(
                algorithm=algorithm,
                learning_rate=0.001,
                discount_factor=0.95,
                exploration_rate=0.1,
                exploration_decay=0.99,
                min_exploration_rate=0.01,
                seed=33,
            )
            restored.load(str(path))
            after = {a.index: v for a, v in restored.get_policy(state).items()}

        assert before.keys() == after.keys()
        for action_idx, value in before.items():
            assert after[action_idx] == pytest.approx(value, rel=1e-5, abs=1e-5)


if __name__ == "__main__":
    unittest.main()
